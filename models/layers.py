import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from models.attention import MLA_Attention

class RMSNorm(nn.Module):
    """DeepSeek-V3 style RMSNorm for improved training stability."""
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = torch.mean(x.pow(2), dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return self.weight * x_normed

class PositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding for sequence awareness."""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, start_pos:start_pos + seq_len]

class QuantCB_FFN(nn.Module):
    """Standard Feed-Forward Network with controlled initialization."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff, bias=False)
        self.activation = nn.GELU()
        self.w_2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        # --- FIX: INITIALIZATION ---
        # Small weights prevent the 260.0 initial loss explosion
        nn.init.normal_(self.w_1.weight, std=0.02)
        # Residual scaling: keeps the variance of the residual stream stable
        nn.init.normal_(self.w_2.weight, std=0.02 / math.sqrt(2 * 6)) # 6 is n_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w_2(self.activation(self.w_1(x))))

class QuantCB_MoE(nn.Module):
    """Sparse Mixture of Experts with optimized routing."""
    def __init__(self, d_model: int, d_ff: int, num_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        self.router = nn.Linear(d_model, num_experts, bias=False)
        # Start router undecided so l_aux is stable at Step 0
        nn.init.normal_(self.router.weight, std=0.01)
        
        self.experts = nn.ModuleList([
            QuantCB_FFN(d_model, d_ff) for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model) 
        
        router_logits = self.router(x_flat)
        weights = F.softmax(router_logits, dim=-1)
        
        top_k_weights, top_k_indices = torch.topk(weights, self.top_k, dim=-1)
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-6)
        
        # --- AUX LOSS ---
        mean_probs = weights.mean(dim=0)
        expert_mask = F.one_hot(top_k_indices, num_classes=self.num_experts).float()
        density_probs = expert_mask.mean(dim=(0, 1))
        l_aux = self.num_experts * torch.sum(mean_probs * density_probs)
        
        final_output = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            token_idx, top_k_idx = torch.where(top_k_indices == i)
            if token_idx.numel() > 0:
                expert_out = expert(x_flat[token_idx])
                final_output[token_idx] += expert_out * top_k_weights[token_idx, top_k_idx].unsqueeze(-1)
                
        return final_output.view(batch, seq_len, d_model), l_aux

class QuantCB_Block(nn.Module):
    """Transformer block integrating Multi-Head Latent Attention (MLA) and MoE."""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, 
                 latent_dim: int = 128, head_dim: int = 64, 
                 num_experts: int = 8, top_k: int = 2, dropout: float = 0.1):
        super().__init__()
        self.ln_1 = RMSNorm(d_model)
        self.attn = MLA_Attention(d_model, n_heads, latent_dim, head_dim)
        
        self.ln_2 = RMSNorm(d_model)
        self.moe = QuantCB_MoE(d_model, d_ff, num_experts=num_experts, top_k=top_k)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, 
                layer_past: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        
        # 1. MLA Attention
        residual = x
        x_norm = self.ln_1(x)
        attn_out, present = self.attn(x_norm, mask=mask, layer_past=layer_past)
        x = residual + attn_out
        
        # 2. MoE
        residual = x
        x_norm = self.ln_2(x)
        moe_out, l_aux = self.moe(x_norm)
        x = residual + moe_out
        
        return x, present, l_aux