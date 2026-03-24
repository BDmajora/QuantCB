import torch
import torch.nn as nn
import math

class QuantCB_Attention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Key, Query, Value projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # Output projection
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch, seq_len, _ = x.shape

        # 1. Linear Projections & Reshape for multi-head
        # (B, L, D) -> (B, L, H, D_K)
        q = self.W_q(x).view(batch, seq_len, self.n_heads, self.d_k)
        k = self.W_k(x).view(batch, seq_len, self.n_heads, self.d_k)
        v = self.W_v(x).view(batch, seq_len, self.n_heads, self.d_k)

        # 2. Scaled Dot-Product Attention using einsum
        # b=batch, l=query_len, s=key_len, h=heads, d=head_dim
        # We calculate (Q * K^T) / sqrt(d_k)
        energy = torch.einsum("blhd, bshd -> blsh", q, k) / math.sqrt(self.d_k)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # 3. Softmax to get attention weights
        attention = torch.softmax(energy, dim=2)

        # 4. Apply attention to values
        # (B, L, S, H) * (B, S, H, D) -> (B, L, H, D)
        out = torch.einsum("blsh, bshd -> blhd", attention, v)
        
        # 5. Concatenate heads and project output
        out = out.contiguous().view(batch, seq_len, self.d_model)
        return self.W_o(out)