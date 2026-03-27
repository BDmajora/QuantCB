import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List

class QuantCB_Engine(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, idx, targets=None, mask=None, past_key_values=None, start_pos=0):
        batch, seq_len = idx.shape
        device = idx.device
        
        # FIX: Ensure mask is boolean for the ~mask logic in MLA
        if mask is None and seq_len > 1 and past_key_values is None:
            mask = torch.tril(torch.ones(seq_len, seq_len, device=device)) == 1
            mask = mask.view(1, 1, seq_len, seq_len)

        # FIX: Removed sqrt scaling and absolute positional encoding
        x = self.model.token_embedding(idx)
        
        presents = [] if past_key_values is None else past_key_values
        new_presents = []
        total_aux_loss = 0.0
        
        for i, block in enumerate(self.model.blocks):
            layer_past = presents[i] if past_key_values is not None else None
            # The MLA block inside here now handles RoPE internally
            x, present, l_aux = block(x, mask=mask, layer_past=layer_past)
            new_presents.append(present)
            total_aux_loss += l_aux
            
        x = self.model.ln_f(x)
        logits = self.model.lm_head(x) 
        
        loss = None
        if targets is not None:
            # NTP Loss
            loss_main = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1)
            )
            
            # MTP Logic
            if seq_len > 1:
                h_n = x[:, :-1, :]
                target_next_plus_one = targets[:, 1:] 
                logits_mtp, _ = self.model.mtp(h_n, targets[:, :-1])
                
                loss_mtp = F.cross_entropy(
                    logits_mtp.reshape(-1, logits_mtp.size(-1)),
                    target_next_plus_one.reshape(-1)
                )
                
                loss = loss_main + (0.1 * loss_mtp) + (0.01 * total_aux_loss)
            else:
                loss = loss_main + (0.01 * total_aux_loss)
            
        return logits, loss, new_presents

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None, repetition_penalty=1.1):
        self.model.eval()
        past_key_values = None
        
        for _ in range(max_new_tokens):
            if past_key_values is None:
                idx_cond = idx
            else:
                idx_cond = idx[:, -1:]
                
            logits, _, next_past_kv = self.forward(
                idx_cond, 
                past_key_values=past_key_values
            )
            past_key_values = next_past_kv
            
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            
            # Apply penalty
            if repetition_penalty != 1.0:
                for i in range(idx.shape[0]):
                    for token_id in set(idx[i].tolist()):
                        if logits[i, token_id] > 0:
                            logits[i, token_id] /= repetition_penalty
                        else:
                            logits[i, token_id] *= repetition_penalty

            # Top-P Sampling
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                for i in range(idx.shape[0]):
                    indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
                    logits[i, indices_to_remove] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx