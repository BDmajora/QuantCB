import torch
import torch.nn as nn
import torch.nn.functional as F
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

    def forward(self, x, start_pos=0):
        seq_len = x.size(1)
        return x + self.pe[:, start_pos:start_pos + seq_len]

class QuantCB_FFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.activation = nn.GELU()
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))

class QuantCB_MoE(nn.Module):
    def __init__(self, d_model, d_ff, num_experts=8, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Router determines which experts process which tokens
        self.router = nn.Linear(d_model, num_experts, bias=False)
        
        # Collection of experts (Standard FFNs)
        self.experts = nn.ModuleList([
            QuantCB_FFN(d_model, d_ff) for _ in range(num_experts)
        ])

    def forward(self, x):
        batch, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model) 
        
        # 1. Get routing weights
        router_logits = self.router(x_flat)
        weights = F.softmax(router_logits, dim=-1)
        
        # 2. Select Top-K experts
        top_k_weights, top_k_indices = torch.topk(weights, self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        # 3. Sparse execution (Dispatching tokens to assigned experts)
        out = torch.zeros_like(x_flat)
        for i in range(self.num_experts):
            token_indices, expert_rank = torch.where(top_k_indices == i)
            if token_indices.numel() > 0:
                expert_out = self.experts[i](x_flat[token_indices])
                out[token_indices] += expert_out * top_k_weights[token_indices, expert_rank].unsqueeze(-1)
                
        return out.view(batch, seq_len, d_model)

class QuantCB_Block(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, latent_dim=128, head_dim=64, num_experts=8, top_k=2, dropout=0.1):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = MLA_Attention(d_model, n_heads, latent_dim, head_dim)
        
        self.ln_2 = nn.LayerNorm(d_model)
        
        # Initialized MoE instead of a single FFN
        self.moe = QuantCB_MoE(d_model, d_ff, num_experts=num_experts, top_k=top_k)

    def forward(self, x, mask=None, layer_past=None):
        # 1. Communication (MLA)
        attn_out, present = self.attn(self.ln_1(x), mask=mask, layer_past=layer_past)
        x = x + attn_out
        
        # 2. Computation (Sparse Expert Activation)
        x = x + self.moe(self.ln_2(x))
        
        return x, present