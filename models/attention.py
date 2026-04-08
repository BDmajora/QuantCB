import torch
import torch.nn.functional as F
import math
from typing import Optional, Tuple

# Changed from .rope to rope for direct script execution
try:
    from .rope import apply_rotary_pos_emb
except ImportError:
    from rope import apply_rotary_pos_emb 

def dynamic_ntk_rope_stateless(seq_len: int, head_dim: int, inv_freq: torch.Tensor, max_position_embeddings: int = 256) -> Tuple[torch.Tensor, torch.Tensor]:
    seq = torch.arange(seq_len, device=inv_freq.device, dtype=inv_freq.dtype)
    freqs = torch.einsum("i,j->ij", seq, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]

def mla_attention_stateless(
    x: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    layer_past: Optional[torch.Tensor] = None,
    wq_weight: torch.Tensor = None,
    wdkv_weight: torch.Tensor = None,
    ln_kv_weight: torch.Tensor = None,
    ln_kv_bias: torch.Tensor = None,
    wuk_weight: torch.Tensor = None,
    wuv_weight: torch.Tensor = None,
    wo_weight: torch.Tensor = None,
    rope_inv_freq: torch.Tensor = None,
    n_heads: int = 8,
    latent_dim: int = 128,
    head_dim: int = 64
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size, seq_len, _ = x.shape
    qk_rope_dim = head_dim // 2
    qk_nope_dim = head_dim - qk_rope_dim
    
    q = F.linear(x, wq_weight).view(batch_size, seq_len, n_heads, head_dim)
    q_nope, q_pe = q.split([qk_nope_dim, qk_rope_dim], dim=-1)
    q_nope = q_nope.transpose(1, 2)
    q_pe = q_pe.transpose(1, 2)
    
    c_kv_proj = F.linear(x, wdkv_weight)
    c_kv = F.layer_norm(c_kv_proj, (latent_dim,), weight=ln_kv_weight, bias=ln_kv_bias)
    
    if layer_past is not None:
        c_kv = torch.cat([layer_past, c_kv], dim=1)
    present_c_kv = c_kv 
    
    full_seq_len = c_kv.size(1)
    k = F.linear(c_kv, wuk_weight).view(batch_size, full_seq_len, n_heads, head_dim)
    v = F.linear(c_kv, wuv_weight).view(batch_size, full_seq_len, n_heads, head_dim)
    
    k_nope, k_pe = k.split([qk_nope_dim, qk_rope_dim], dim=-1)
    k_nope = k_nope.transpose(1, 2)
    k_pe = k_pe.transpose(1, 2)
    v = v.transpose(1, 2)
    
    cos, sin = dynamic_ntk_rope_stateless(full_seq_len, qk_rope_dim, rope_inv_freq)
    
    q_pe, _ = apply_rotary_pos_emb(q_pe, q_pe, cos[:, :, -seq_len:, :], sin[:, :, -seq_len:, :])
    _, k_pe = apply_rotary_pos_emb(k_pe, k_pe, cos, sin)
    
    q_final = torch.cat([q_nope, q_pe], dim=-1)
    k_final = torch.cat([k_nope, k_pe], dim=-1)
    
    attn_scores = torch.matmul(q_final, k_final.transpose(-2, -1)) / math.sqrt(head_dim)
    
    if mask is not None:
        if mask.dtype == torch.bool:
            attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
        else:
            attn_scores = attn_scores + mask
        
    attn_weights = F.softmax(attn_scores, dim=-1)
    context = torch.matmul(attn_weights, v)
    context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
    
    out = F.linear(context, wo_weight)
    return out, present_c_kv