import torch
import torch.nn as nn
import math

class QuantCB_Attention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch, seq_len, _ = x.shape

        # 1. Linear Projections
        q = self.W_q(x).view(batch, seq_len, self.n_heads, self.d_k)
        k = self.W_k(x).view(batch, seq_len, self.n_heads, self.d_k)
        v = self.W_v(x).view(batch, seq_len, self.n_heads, self.d_k)

        # 2. Scaled Dot-Product: (B, L, H, D) x (B, S, H, D) -> (B, H, L, S)
        scores = torch.einsum("bqhd, bkhd -> bhqk", q, k) / math.sqrt(self.d_k)

        # 3. Causal Masking (The "Graph Integrity" Step)
        # Create lower-triangular mask of 1s
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
        # Expand to match score dimensions (Batch, Heads, L, S)
        mask = mask.view(1, 1, seq_len, seq_len)
        # Fill zeros (future) with -inf
        scores = scores.masked_fill(mask == 0, float("-inf"))

        # 4. Softmax & Value Weighting
        attn = torch.softmax(scores, dim=-1)
        out = torch.einsum("bhqk, bkhd -> bqhd", attn, v)
        
        # 5. Output Projection
        out = out.contiguous().view(batch, seq_len, self.d_model)
        return self.W_o(out)