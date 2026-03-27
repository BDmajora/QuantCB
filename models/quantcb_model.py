import torch
import torch.nn as nn
import math
from models.layers import QuantCB_Block, PositionalEncoding

class QuantCB_Model(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=8, d_ff=1024, n_layers=6, latent_dim=128, head_dim=64, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        
        self.blocks = nn.ModuleList([
            QuantCB_Block(d_model, n_heads, d_ff, latent_dim, head_dim, dropout) 
            for _ in range(n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self.token_embedding.weight = self.lm_head.weight

    def forward(self, idx, targets=None, mask=None, past_key_values=None, start_pos=0):
        batch, seq_len = idx.shape
        device = idx.device
        
        # Only apply causal mask if processing a sequence > 1 without cache
        if mask is None and seq_len > 1 and past_key_values is None:
            mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).view(1, 1, seq_len, seq_len)
        elif past_key_values is not None:
            mask = None # Single token generation doesn't need masking against the past

        x = self.token_embedding(idx) * math.sqrt(self.d_model)
        x = self.pos_encoding(x, start_pos=start_pos)
        
        presents = [] if past_key_values is None else past_key_values
        new_presents = []
        
        for i, block in enumerate(self.blocks):
            layer_past = presents[i] if past_key_values is not None else None
            x, present = block(x, mask=mask, layer_past=layer_past)
            new_presents.append(present)
            
        x = self.ln_f(x)
        logits = self.lm_head(x) 
        
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1)
            )
            
        return logits, loss, new_presents

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Optimized generation utilizing the MLA Latent Cache.
        """
        self.eval()
        past_key_values = None
        
        for i in range(max_new_tokens):
            if past_key_values is None:
                # First step: process the full initial context
                idx_cond = idx
                start_pos = 0
            else:
                # Subsequent steps: process ONLY the last generated token
                idx_cond = idx[:, -1:]
                start_pos = idx.size(1) - 1
                
            # Forward pass returning the updated cache
            logits, _, past_key_values = self(
                idx_cond, 
                past_key_values=past_key_values,
                start_pos=start_pos
            )
            
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            idx = torch.cat((idx, idx_next), dim=1)

        return idx