import torch
import torch.nn as nn
import math
from models.attention import MLA_Attention

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class QuantCB_FFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.activation = nn.GELU()
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))

class QuantCB_Block(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, latent_dim=128, head_dim=64, dropout=0.1):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        # Swapped QuantCB_Attention for MLA_Attention
        self.attn = MLA_Attention(d_model, n_heads, latent_dim, head_dim)
        
        self.ln_2 = nn.LayerNorm(d_model)
        self.ffn = QuantCB_FFN(d_model, d_ff, dropout)

    def forward(self, x, mask=None):
        # Communication Sub-layer (MLA)
        x = x + self.attn(self.ln_1(x), mask=mask)
        # Computation Sub-layer (FFN)
        x = x + self.ffn(self.ln_2(x))
        return x