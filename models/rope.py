import torch
from typing import Tuple

def compute_ntk_inv_freq(dim: int, seq_len: int, max_position_embeddings: int, base: float, device: torch.device) -> torch.Tensor:
    """Helper to compute NTK-scaled frequencies without mutating module state."""
    if seq_len > max_position_embeddings:
        base = base * ((seq_len / max_position_embeddings) - 0.5) ** (dim / (dim - 2))
    
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    return inv_freq

def compute_rope_cache_stateless(
    seq_len: int, 
    dim: int, 
    device: torch.device, 
    dtype: torch.dtype,
    max_position_embeddings: int = 2048,
    base: float = 10000.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pure functional RoPE cache generation.
    Replaces DynamicNTKRotaryEmbedding class for AOT compilation.
    """
    inv_freq = compute_ntk_inv_freq(dim, seq_len, max_position_embeddings, base, device)
    
    t = torch.arange(seq_len, device=device, dtype=dtype)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1) # Shape: (Seq, Dim)
    
    cos = emb.cos()[None, None, :, :] # (1, 1, Seq, Dim)
    sin = emb.sin()[None, None, :, :] # (1, 1, Seq, Dim)
    return cos, sin

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pure functional RoPE application.
    FIXED: Slices cos/sin to match input dimension to prevent 32 vs 64 errors.
    """
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    
    # --- CRITICAL FIX ---
    # In MLA, head_dim might be 64, but the RoPE part is only 32.
    # We ensure the cos/sin match the last dimension of q/k.
    dim = q.shape[-1]
    cos_active = cos[..., :dim]
    sin_active = sin[..., :dim]
    
    q_embed = (q * cos_active) + (rotate_half(q) * sin_active)
    k_embed = (k * cos_active) + (rotate_half(k) * sin_active)
    return q_embed, k_embed