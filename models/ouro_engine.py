import torch
import torch.nn as nn
import torch.nn.functional as F

class Ouro_Engine(nn.Module):
    def __init__(self, model, max_loops=4, exit_threshold=0.5):
        super().__init__()
        self.model = model
        self.max_loops = max_loops
        self.exit_threshold = exit_threshold
        
        # Phase 2: Latent Probe
        self.latent_probe = nn.Linear(model.token_embedding.embedding_dim, 1)
        
        # Phase 2: Weighted Residual Connection
        self.thinking_gate = nn.Parameter(torch.tensor([0.0]))

    def forward(self, idx, targets=None, mask=None, past_key_values=None, spec_threshold=None):
        batch, seq_len = idx.shape
        device = idx.device
        
        if mask is None and seq_len > 1 and past_key_values is None:
            mask = torch.tril(torch.ones(seq_len, seq_len, device=device)) == 1
            mask = mask.view(1, 1, seq_len, seq_len)

        x = self.model.token_embedding(idx)
        loop_x = x 
        
        all_probe_logits = [] 
        total_aux_loss = torch.tensor(0.0, device=device)
        gate = torch.sigmoid(self.thinking_gate)

        # --- Recurrent Thinking Loop ---
        for loop_idx in range(self.max_loops):
            prev_x = loop_x.clone() 
            current_x = prev_x
            for block in self.model.blocks:
                current_x, _, l_aux = block(current_x, mask=mask, layer_past=None)
                total_aux_loss = total_aux_loss + l_aux
            
            loop_x = (1.0 - gate) * prev_x + gate * current_x
            loop_x_norm = self.model.ln_f(loop_x)
            all_probe_logits.append(self.latent_probe(loop_x_norm))
            
            if not self.training and targets is None and seq_len == 1:
                logits_step = self.model.lm_head(loop_x_norm)
                probs = F.softmax(logits_step[:, -1, :], dim=-1)
                entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1).mean()
                if entropy < (spec_threshold or self.exit_threshold):
                    break

        # Final pass for standard logits
        final_norm = self.model.ln_f(loop_x)
        logits = self.model.lm_head(final_norm)

        if targets is not None:
            # 1. Main Language Modeling Loss
            loss_main = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
            # 2. MTP Loss (Updated for Functional Weight Sharing)
            loss_mtp = torch.tensor(0.0, device=device)
            if seq_len > 1:
                # Prepare base hidden states (all but the last)
                h_n = final_norm[:, :-1, :]
                # Prepare target labels used for embedding lookup (all but the last)
                target_ids_for_mtp = targets[:, :-1]
                
                # STEP A: Generate embeddings functionally from the SHARED layer
                x_embed_mtp = self.model.token_embedding(target_ids_for_mtp)
                
                # STEP B: Run MTP module to get MTP hidden states
                h_mtp = self.model.mtp(h_n, x_embed_mtp)
                
                # STEP C: Project to logits using the SHARED LM head
                logits_mtp = self.model.lm_head(h_mtp)
                
                # STEP D: Calculate loss against shifted targets (predicting n+1)
                loss_mtp = F.cross_entropy(
                    logits_mtp.reshape(-1, logits_mtp.size(-1)), 
                    targets[:, 1:].reshape(-1)
                )
            
            loss = loss_main + (0.1 * loss_mtp) + (0.01 * total_aux_loss)
            return logits, loss, all_probe_logits
            
        return logits, None, None

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, **kwargs):
        # Implementation remains largely the same, ensuring it uses the loop above.
        self.model.eval()
        for _ in range(max_new_tokens):
            logits, _, _ = self.forward(idx, **kwargs)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx