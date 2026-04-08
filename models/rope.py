import torch
import torch.nn as nn
from typing import Tuple

def compute_ntk_inv_freq(dim: int, seq_len: int, max_position_embeddings: int, base: float, device: torch.device) -> torch.Tensor:
    """Helper to compute NTK-scaled frequencies without mutating module state."""
    # Dynamic NTK scaling formula
    if seq_len > max_position_embeddings:
        base = base * ((seq_len / max_position_embeddings) - 0.5) ** (dim / (dim - 2))
    
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
    return inv_freq

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pure functional RoPE application."""
    def rotate_half(x):
        # We use the standard chunk and flip for RoPE
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    
    # Broadcast cos/sin and apply the transformation
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class DynamicNTKRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_position_embeddings = max_position_embeddings
        
        # Initial frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=True)
        
        # We store these as attributes, not buffers, to avoid 
        # graph mutations during the forward pass.
        self.cos_cached = None
        self.sin_cached = None
        self.max_seq_len_cached = -1

    def _update_cache(self, seq_len, device, dtype):
        """Update cos/sin tensors only when needed."""
        # Re-compute inv_freq based on sequence length (Dynamic NTK)
        inv_freq = compute_ntk_inv_freq(
            self.dim, seq_len, self.max_position_embeddings, self.base, device
        )
        
        t = torch.arange(seq_len, device=device, dtype=dtype)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self.cos_cached = emb.cos()[None, None, :, :].to(dtype)
        self.sin_cached = emb.sin()[None, None, :, :].to(dtype)
        self.max_seq_len_cached = seq_len

    def forward(self, x, seq_len):
        # If cache is too small or device changed, update it
        if seq_len > self.max_seq_len_cached or self.cos_cached is None or self.cos_cached.device != x.device:
            self._update_cache(seq_len, x.device, x.dtype)
            
        return self.cos_cached[:, :, :seq_len, ...], self.sin_cached[:, :, :seq_len, ...]