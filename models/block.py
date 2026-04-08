import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

# Guarded imports to handle both relative (module) and absolute (standalone) paths
try:
    from .attention import mla_attention_stateless
    from .moe import quantcb_moe_stateless
except (ImportError, ValueError):
    from attention import mla_attention_stateless
    from moe import quantcb_moe_stateless

def rms_norm_stateless(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Pure functional implementation of RMSNorm.
    Matches the RDNA2/SPIR-V expected math for normalization.
    """
    # Calculation: x * 1/sqrt(mean(x^2) + eps) * weight
    variance = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * weight

def quantcb_block_forward_stateless(
    x: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    layer_past: Optional[torch.Tensor] = None,
    # --- Shared Block Weights ---
    ln_1_weight: torch.Tensor = None,
    ln_2_weight: torch.Tensor = None,
    # --- Sub-Module Weights ---
    # These dictionaries MUST have keys that match the function signatures exactly
    attn_weights: Dict[str, torch.Tensor] = None,
    moe_weights: Dict[str, torch.Tensor] = None,
    # --- Architecture Config ---
    n_heads: int = 8,
    latent_dim: int = 128,
    head_dim: int = 64,
    num_experts: int = 8,
    top_k: int = 2
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """
    Pure functional QuantCB Block pass for Vulkan/IREE compilation.
    Bypasses PyTorch class state entirely to ensure memory buffer alignment.
    """
    
    # 1. Attention Path (Residual 1)
    residual_1 = x
    
    # Stateless RMSNorm 1: Preparing the stream for MLA
    x_norm1 = rms_norm_stateless(x, ln_1_weight)
    
    # Functional MLA Attention call
    # **attn_weights unpacks: wq_weight, wdkv_weight, wuk_weight, wuv_weight, wo_weight, etc.
    attn_out, present = mla_attention_stateless(
        x_norm1, 
        mask=mask, 
        layer_past=layer_past,
        n_heads=n_heads,
        latent_dim=latent_dim,
        head_dim=head_dim,
        **attn_weights 
    )
    x = residual_1 + attn_out
    
    # 2. MoE Path (Residual 2)
    residual_2 = x
    
    # Stateless RMSNorm 2: Preparing the stream for SwiGLU experts
    x_norm2 = rms_norm_stateless(x, ln_2_weight)
    
    # Functional QuantCB MoE call
    # **moe_weights unpacks: router_weight, w1_weight, w2_weight, w3_weight
    moe_out, l_aux = quantcb_moe_stateless(
        x_norm2,
        num_experts=num_experts,
        top_k=top_k,
        **moe_weights
    )
    x = residual_2 + moe_out
    
    return x, present, l_aux