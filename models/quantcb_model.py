import torch
import torch.nn as nn
import math
from models.layers import QuantCB_Block, PositionalEncoding

class QuantCB_Model(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=8, d_ff=1024, n_layers=6, latent_dim=128, head_dim=64, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # 1. Input: Token IDs -> Continuous Vectors
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        
        # 2. Backbone: Stacked Transformer Blocks using MLA
        self.blocks = nn.ModuleList([
            QuantCB_Block(d_model, n_heads, d_ff, latent_dim, head_dim, dropout) 
            for _ in range(n_layers)
        ])
        
        # 3. Output Head: Hidden States -> Vocab Probabilities
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight Tying (Saves ~50MB of VRAM for 50k vocab)
        self.token_embedding.weight = self.lm_head.weight

    def forward(self, idx, targets=None, mask=None):
        batch, seq_len = idx.shape
        device = idx.device
        
        # Internal Causal Masking (strictly prohibits looking into the future)
        if mask is None:
            mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).view(1, 1, seq_len, seq_len)

        # Forward through embeddings and encoding
        # Scaling by sqrt(d_model) stabilizes gradients for deeper stacks
        x = self.token_embedding(idx) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        # Forward through the block stack with masking
        for block in self.blocks:
            x = block(x, mask=mask)
            
        x = self.ln_f(x)
        logits = self.lm_head(x) # (Batch, Seq, Vocab_Size)
        
        loss = None
        if targets is not None:
            # Flatten tensors for CrossEntropyLoss (Batch*Seq, Vocab_Size)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1)
            )
            
        return logits, loss