import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List

class Ouro_Engine(nn.Module):
    def __init__(self, model, max_loops=4, exit_threshold=0.5):
        super().__init__()
        self.model = model
        self.max_loops = max_loops
        self.exit_threshold = exit_threshold
        
        # Phase 2: Latent Probe for "Clean vs Corrupted" supervision
        self.latent_probe = nn.Linear(model.token_embedding.embedding_dim, 1)
        
        # Phase 2: Weighted Residual Connection ("Thinking Gate")
        # Starts at 0.0, which means torch.sigmoid(0.0) = 0.5 (equal mix)
        self.thinking_gate = nn.Parameter(torch.tensor([0.0]))

    def forward(self, idx, targets=None, mask=None, past_key_values=None, spec_threshold=None, hallucination_tags=None):
        batch, seq_len = idx.shape
        device = idx.device
        
        # Ensure mask is boolean for MLA/Attention logic
        if mask is None and seq_len > 1 and past_key_values is None:
            mask = torch.tril(torch.ones(seq_len, seq_len, device=device)) == 1
            mask = mask.view(1, 1, seq_len, seq_len)

        # Start with base embeddings
        x = self.model.token_embedding(idx)
        
        # FIX: Move loop_x outside so the model "builds" on its thoughts across loops
        loop_x = x 
        
        presents = [] if past_key_values is None else past_key_values
        logits = None
        loss = None
        all_probe_logits = [] 
        
        threshold = spec_threshold if spec_threshold is not None else self.exit_threshold

        # --- Phase 2: Shared-Weight Recursion Loop ---
        for loop_idx in range(self.max_loops):
            prev_x = loop_x
            new_presents = []
            total_aux_loss = 0.0
            
            # Learnable gate for weighted residual connection
            gate = torch.sigmoid(self.thinking_gate)
            
            # Process through the Transformer Blocks (with MLA + RoPE)
            current_x = loop_x
            for i, block in enumerate(self.model.blocks):
                layer_past = presents[i] if past_key_values is not None else None
                current_x, present, l_aux = block(current_x, mask=mask, layer_past=layer_past)
                
                # Update KV cache only on the final refined step (Memory Optimization)
                if loop_idx == self.max_loops - 1:
                    new_presents.append(present)
                    
                total_aux_loss += l_aux
            
            # --- WEIGHTED RESIDUAL: The "Thinking" Step ---
            # loop_x evolves by mixing the previous thought with the new calculation
            loop_x = (1 - gate) * prev_x + gate * current_x
                
            loop_x_norm = self.model.ln_f(loop_x)
            logits = self.model.lm_head(loop_x_norm)
            
            # Calculate and store probe logits for supervision
            probe_step = self.latent_probe(loop_x_norm)
            all_probe_logits.append(probe_step)
            
            # --- Entropy Exit Gate (Inference only) ---
            if targets is None and seq_len == 1:
                probs = F.softmax(logits[:, -1, :], dim=-1)
                entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1).mean()
                
                draft_token = torch.argmax(probs, dim=-1).item()
                is_hallucinating = hallucination_tags is not None and draft_token in hallucination_tags
                
                # Exit early if we are confident AND it's not a hallucination token
                if entropy < threshold and not is_hallucinating:
                    break

        # --- LOSS CALCULATION (Training) ---
        if targets is not None:
            # Main Next Token Prediction Loss
            loss_main = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1)
            )
            
            # Multi-Token Prediction (MTP) Logic
            if seq_len > 1:
                h_n = loop_x_norm[:, :-1, :]
                target_next_plus_one = targets[:, 1:] 
                logits_mtp, _ = self.model.mtp(h_n, targets[:, :-1])
                
                loss_mtp = F.cross_entropy(
                    logits_mtp.reshape(-1, logits_mtp.size(-1)),
                    target_next_plus_one.reshape(-1)
                )
                
                loss = loss_main + (0.1 * loss_mtp) + (0.01 * total_aux_loss)
            else:
                loss = loss_main + (0.01 * total_aux_loss)
            
            return logits, loss, all_probe_logits
            
        # Return results and updated KV cache for generation
        return logits, None, new_presents

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, **kwargs):
        self.model.eval()
        past_key_values = None
        
        for _ in range(max_new_tokens):
            idx_cond = idx if past_key_values is None else idx[:, -1:]
                
            # Forward pass provides next_past_kv as the 3rd return
            logits, _, next_past_kv = self.forward(
                idx_cond, 
                past_key_values=past_key_values,
                **kwargs
            )
            past_key_values = next_past_kv
            
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx