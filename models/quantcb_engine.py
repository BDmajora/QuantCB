import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class QuantCB_Engine(nn.Module):
    """
    Handles execution logic, loss calculation, and MTP strategy 
    decoupled from the raw model architecture.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, idx, targets=None, mask=None, past_key_values=None, start_pos=0):
        batch, seq_len = idx.shape
        device = idx.device
        
        # 1. Causal Masking Logic
        if mask is None and seq_len > 1 and past_key_values is None:
            mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).view(1, 1, seq_len, seq_len)
        elif past_key_values is not None:
            mask = None 

        # 2. Extract Embeddings and Positional Encoding
        x = self.model.token_embedding(idx) * math.sqrt(self.model.d_model)
        x = self.model.pos_encoding(x, start_pos=start_pos)
        
        presents = [] if past_key_values is None else past_key_values
        new_presents = []
        
        # 3. Transformer Blocks Execution
        for i, block in enumerate(self.model.blocks):
            layer_past = presents[i] if past_key_values is not None else None
            x, present = block(x, mask=mask, layer_past=layer_past)
            new_presents.append(present)
            
        x = self.model.ln_f(x)
        logits = self.model.lm_head(x) 
        
        # 4. Loss Calculation (Multi-Objective)
        loss = None
        if targets is not None:
            # A. Main next-token prediction loss (t -> t+1)
            loss_main = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1)
            )
            
            # B. MoE Auxiliary Loss (Load Balancing)
            # We crawl the blocks to find the l_aux values we calculated in QuantCB_MoE
            loss_aux = 0.0
            for block in self.model.blocks:
                if hasattr(block.moe, 'l_aux'):
                    loss_aux += block.moe.l_aux
            
            # C. MTP logic: Predict token n+2 (Lookahead Task)
            loss_mtp = 0.0
            if seq_len > 1:
                h_n = x[:, :-1, :]
                hint_tokens = targets[:, :-1]
                
                # Predict token at t+2
                logits_mtp, _ = self.model.mtp(h_n, hint_tokens)
                
                loss_mtp = F.cross_entropy(
                    logits_mtp.reshape(-1, logits_mtp.size(-1)),
                    targets[:, 1:].reshape(-1)
                )
                
                # Combined Loss:
                # 1.0 * Main + 0.1 * MTP (DeepSeek V3 ratio) + 0.01 * MoE Aux
                loss = loss_main + (0.1 * loss_mtp) + (0.01 * loss_aux)
            else:
                # Fallback for short sequences
                loss = loss_main + (0.01 * loss_aux)
            
        return logits, loss, new_presents

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Inference loop using the engine's forward logic.
        """
        self.model.eval()
        past_key_values = None
        
        for _ in range(max_new_tokens):
            if past_key_values is None:
                idx_cond = idx
                start_pos = 0
            else:
                idx_cond = idx[:, -1:]
                start_pos = idx.shape[1] - 1
                
            logits, _, past_key_values = self.forward(
                idx_cond, 
                past_key_values=past_key_values,
                start_pos=start_pos
            )
            
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            idx = torch.cat((idx, idx_next), dim=1)

        return idx