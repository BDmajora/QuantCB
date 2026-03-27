import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from models.layers import QuantCB_Block, PositionalEncoding
from models.mtp_module import MTPModule

class QuantCB_Model(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=8, d_ff=1024, n_layers=6, 
                 latent_dim=128, head_dim=64, num_experts=8, top_k=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        
        self.blocks = nn.ModuleList([
            QuantCB_Block(
                d_model=d_model, 
                n_heads=n_heads, 
                d_ff=d_ff, 
                latent_dim=latent_dim, 
                head_dim=head_dim, 
                num_experts=num_experts, 
                top_k=top_k, 
                dropout=dropout
            ) 
            for _ in range(n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Corrected Weight Tying: Head inherits from Embedding
        self.lm_head.weight = self.token_embedding.weight

        # Integrated MTP Module sharing the tied weights
        self.mtp = MTPModule(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            embedding=self.token_embedding,
            head=self.lm_head
        )

    def forward(self, idx, targets=None, mask=None, past_key_values=None, start_pos=0):
        batch, seq_len = idx.shape
        device = idx.device
        
        # Causal mask logic
        if mask is None and seq_len > 1 and past_key_values is None:
            mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).view(1, 1, seq_len, seq_len)
        elif past_key_values is not None:
            mask = None 

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
            # Main next-token prediction loss (t -> t+1)
            loss_main = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1)
            )
            
            # MTP logic: Predict token n+2
            # We only calculate MTP loss if seq_len > 1 to avoid empty slice crashes
            if seq_len > 1:
                # We use x[:, :-1, :] (hidden states for t=0 to t=L-2)
                # and targets[:, :-1] (actual tokens at t=1 to t=L-1)
                # to predict targets[:, 1:] (actual tokens at t=2 to t=L)
                
                # Handling the tuple return: logits_mtp and the hidden state x_mtp
                logits_mtp, _ = self.mtp(x[:, :-1, :], targets[:, :-1])
                
                loss_mtp = F.cross_entropy(
                    logits_mtp.reshape(-1, logits_mtp.size(-1)),
                    targets[:, 1:].reshape(-1)
                )
                
                # Final loss combining both objectives
                loss = loss_main + 0.1 * loss_mtp
            else:
                loss = loss_main
            
        return logits, loss, new_presents

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        self.eval()
        past_key_values = None
        
        for i in range(max_new_tokens):
            if past_key_values is None:
                idx_cond = idx
                start_pos = 0
            else:
                idx_cond = idx[:, -1:]
                start_pos = idx.size(1) - 1
                
            # We don't need targets or MTP loss during generation
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