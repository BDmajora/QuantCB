import torch
import torch.nn.functional as F

def swiglu_ffn_stateless(
    x: torch.Tensor, 
    w1_weight: torch.Tensor, # Gate projection
    w2_weight: torch.Tensor, # Down projection
    w3_weight: torch.Tensor  # Up projection
) -> torch.Tensor:
    """Pure functional SwiGLU FFN (Standard in LLaMA, DeepSeek, etc.)."""
    gate = F.linear(x, w1_weight)
    up = F.linear(x, w3_weight)
    
    # SwiGLU activation: SiLU(gate) * up
    hidden = F.silu(gate) * up
    
    return F.linear(hidden, w2_weight)