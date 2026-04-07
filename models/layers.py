import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

# Import MLA from attention and RoPE utilities from the new rope.py
from .attention import MLA_Attention
from .rope import DynamicNTKRotaryEmbedding, apply_rotary_pos_emb

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = torch.mean(x.pow(2), dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return self.weight * x_normed

class QuantCB_FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff, bias=False)
        self.activation = nn.GELU()
        self.w_2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        # Initialization to prevent early loss spikes
        nn.init.normal_(self.w_1.weight, std=0.02)
        nn.init.normal_(self.w_2.weight, std=0.02 / math.sqrt(2 * 6)) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w_2(self.activation(self.w_1(x))))

class QuantCB_MoE(nn.Module):
    def __init__(self, d_model: int, d_ff: int, num_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(d_model, num_experts, bias=False)
        nn.init.normal_(self.router.weight, std=0.01)
        
        self.experts = nn.ModuleList([
            QuantCB_FFN(d_model, d_ff) for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model) 
        
        # 1. Get Router Scores
        router_logits = self.router(x_flat)
        weights = F.softmax(router_logits, dim=-1)
        
        # 2. Select Top-K Experts
        top_k_weights, top_k_indices = torch.topk(weights, self.top_k, dim=-1)
        
        # 3. Create Static Weight Mask (The "IREE Fix")
        expert_mask = F.one_hot(top_k_indices, num_classes=self.num_experts).float()
        
        static_weights = (top_k_weights.unsqueeze(-1) * expert_mask).sum(dim=1)
        static_weights = static_weights / (static_weights.sum(dim=-1, keepdim=True) + 1e-6)
        
        # 4. Load balancing loss
        mean_probs = weights.mean(dim=0)
        density_probs = static_weights.mean(dim=0)
        l_aux = self.num_experts * torch.sum(mean_probs * density_probs)
        
        # 5. Static Aggregation Loop
        # Explicit device and dtype binding is required to stop lifts
        final_output = torch.zeros_like(x_flat, device=x_flat.device, dtype=x_flat.dtype)
        
        for i, expert in enumerate(self.experts):
            expert_out = expert(x_flat)
            weight_i = static_weights[:, i].unsqueeze(-1)
            final_output = final_output + (expert_out * weight_i)
                
        return final_output.view(batch, seq_len, d_model), l_aux

class QuantCB_Block(nn.Module):
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
        
        # 1. Attention Path
        residual = x
        attn_out, present = self.attn(self.ln_1(x), mask=mask, layer_past=layer_past)
        x = residual + attn_out
        
        # 2. MoE Path
        residual = x
        moe_out, l_aux = self.moe(self.ln_2(x))
        x = residual + moe_out
        
        return x, present, l_aux