import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

# Guarded imports
try:
    from .attention import mla_attention_stateless
    from .moe import quantcb_moe_stateless
    from .rope import compute_rope_cache_stateless
except (ImportError, ValueError):
    from attention import mla_attention_stateless
    from moe import quantcb_moe_stateless
    from rope import compute_rope_cache_stateless

def rms_norm_stateless(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Stateless RMSNorm for RDNA2/SPIR-V."""
    variance = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * weight

def quantcb_block_forward_stateless(
    x: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    layer_past: Optional[torch.Tensor] = None,
    ln_1_weight: torch.Tensor = None,
    ln_2_weight: torch.Tensor = None,
    attn_weights: Dict[str, torch.Tensor] = None,
    moe_weights: Dict[str, torch.Tensor] = None,
    n_heads: int = 8,
    latent_dim: int = 128,
    head_dim: int = 64,
    num_experts: int = 8,
    top_k: int = 2
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """
    Pure functional QuantCB Block.
    No class state. Optimized for Vulkan memory buffers.
    """
    batch, seq_len, _ = x.shape
    
    # 1. Attention Path
    residual = x
    x_norm1 = rms_norm_stateless(x, ln_1_weight)
    
    # --- FIXED: Generate RoPE Cache Statelessly here if not passed in ---
    # Most MLA implementations use 1/2 of head_dim for RoPE (e.g. 32)
    rope_dim = head_dim // 2 
    cos, sin = compute_rope_cache_stateless(
        seq_len=seq_len + (layer_past.shape[2] if layer_past is not None else 0),
        dim=rope_dim,
        device=x.device,
        dtype=x.dtype
    )
    
    attn_out, present = mla_attention_stateless(
        x_norm1, 
        mask=mask, 
        layer_past=layer_past,
        cos=cos,
        sin=sin,
        n_heads=n_heads,
        latent_dim=latent_dim,
        head_dim=head_dim,
        **attn_weights 
    )
    x = residual + attn_out
    
    # 2. MoE Path
    residual = x
    x_norm2 = rms_norm_stateless(x, ln_2_weight)
    
    moe_out, l_aux = quantcb_moe_stateless(
        x_norm2,
        num_experts=num_experts,
        top_k=top_k,
        **moe_weights
    )
    x = residual + moe_out
    
    return x, present, l_aux