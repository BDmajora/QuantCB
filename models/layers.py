import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.attention import MLA_Attention

class RMSNorm(nn.Module):
    """DeepSeek-V3 style RMSNorm for improved training stability."""
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        norm_x = x.pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return self.weight * x_normed

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
        # FIX: Activation -> Projection -> Dropout (Standard Transformer Order)
        return self.dropout(self.w_2(self.activation(self.w_1(x))))

class QuantCB_MoE(nn.Module):
    def __init__(self, d_model, d_ff, num_experts=8, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList([
            QuantCB_FFN(d_model, d_ff) for _ in range(num_experts)
        ])
        
        # Buffer to store auxiliary loss for the engine to collect
        self.l_aux = 0.0

    def forward(self, x):
        batch, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model) 
        
        # 1. Get routing weights
        router_logits = self.router(x_flat)
        weights = F.softmax(router_logits, dim=-1)
        
        # 2. Select Top-K experts
        top_k_weights, top_k_indices = torch.topk(weights, self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        # --- AUX LOSS CALCULATION (Prevents Router Collapse) ---
        # Mean probability assigned to each expert by the router
        mean_probs = weights.mean(dim=0)
        # Fraction of tokens actually sent to each expert
        expert_counts = torch.bincount(top_k_indices.flatten(), minlength=self.num_experts)
        routing_fraction = expert_counts.float() / top_k_indices.numel()
        # The penalty: Encourages uniform distribution
        self.l_aux = self.num_experts * torch.sum(mean_probs * routing_fraction)
        
        # --- AUTOGRAD-SAFE EXECUTION ---
        # Use torch.zeros to avoid inheriting old gradients from x_flat during in-place ops
        out = torch.zeros(x_flat.shape, dtype=x_flat.dtype, device=x_flat.device)
        
        for i in range(self.num_experts):
            token_indices, expert_rank = torch.where(top_k_indices == i)
            if token_indices.numel() > 0:
                expert_out = self.experts[i](x_flat[token_indices])
                # Scale expert output by the router weight
                out[token_indices] += expert_out * top_k_weights[token_indices, expert_rank].unsqueeze(-1)
                
        return out.view(batch, seq_len, d_model)

class QuantCB_Block(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, latent_dim=128, head_dim=64, num_experts=8, top_k=2, dropout=0.1):
        super().__init__()
        # Swapped to RMSNorm for MoE stability
        self.ln_1 = RMSNorm(d_model)
        self.attn = MLA_Attention(d_model, n_heads, latent_dim, head_dim)
        
        self.ln_2 = RMSNorm(d_model)
        self.moe = QuantCB_MoE(d_model, d_ff, num_experts=num_experts, top_k=top_k)

    def forward(self, x, mask=None, layer_past=None):
        # 1. MLA Attention with Residual Connection
        attn_out, present = self.attn(self.ln_1(x), mask=mask, layer_past=layer_past)
        x = x + attn_out
        
        # 2. MoE Computation with Residual Connection
        x = x + self.moe(self.ln_2(x))
        
        return x, present