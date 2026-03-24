import torch
import torch.nn as nn
import math
from models.attention import QuantCB_Attention

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # Create a matrix of [max_len, d_model] representing positions
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Log-space frequency computation for numerical stability
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Fill sine for even indices, cosine for odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (stays on GPU but not updated by optimizer)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x shape: (Batch, Seq_Len, d_model)
        # Add encoding to embeddings up to the current sequence length
        return x + self.pe[:, :x.size(1)]

class QuantCB_FFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        # Expansion layer (usually 4x d_model)
        self.w_1 = nn.Linear(d_model, d_ff)
        # SOTA activation for Transformer architectures
        self.activation = nn.GELU()
        # Contraction layer back to model dimension
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))

class QuantCB_Block(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        # Pre-Norm components
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = QuantCB_Attention(d_model, n_heads)
        
        self.ln_2 = nn.LayerNorm(d_model)
        self.ffn = QuantCB_FFN(d_model, d_ff, dropout)

    def forward(self, x):
        # Communication Sub-layer with Residual Connection
        x = x + self.attn(self.ln_1(x))
        # Computation Sub-layer with Residual Connection
        x = x + self.ffn(self.ln_2(x))
        return x