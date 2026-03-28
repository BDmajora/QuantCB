import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

# Import the Dynamic RoPE components from your layers file
from .rope import DynamicNTKRotaryEmbedding, apply_rotary_pos_emb

class MLA_Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, latent_dim: int = 128, head_dim: int = 64):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.latent_dim = latent_dim
        
        # Split head_dim: half for content, half for rotary positional info
        self.qk_rope_dim = head_dim // 2 
        self.qk_nope_dim = head_dim - self.qk_rope_dim
        
        # Projections
        self.W_q = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.W_dkv = nn.Linear(d_model, latent_dim, bias=False)
        self.ln_kv = nn.LayerNorm(latent_dim)
        
        self.W_uk = nn.Linear(latent_dim, n_heads * head_dim, bias=False)
        self.W_uv = nn.Linear(latent_dim, n_heads * head_dim, bias=False)
        self.W_o = nn.Linear(n_heads * head_dim, d_model, bias=False)

        # Dynamic RoPE Scaling (NTK-aware)
        self.rope = DynamicNTKRotaryEmbedding(self.qk_rope_dim, max_position_embeddings=256)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, 
                layer_past: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        
        # 1. Generate Queries and split into Nope/Rope
        q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        q_nope, q_pe = q.split([self.qk_nope_dim, self.qk_rope_dim], dim=-1)
        q_nope = q_nope.transpose(1, 2)
        q_pe = q_pe.transpose(1, 2)
        
        # 2. Compress current tokens to Latent Vector
        c_kv = self.ln_kv(self.W_dkv(x)) 
        
        # 3. KV Cache Update
        if layer_past is not None:
            c_kv = torch.cat([layer_past, c_kv], dim=1)
        present_c_kv = c_kv 
        
        # 4. Expand Latent Vector to Keys and Values
        full_seq_len = c_kv.size(1)
        k = self.W_uk(c_kv).view(batch_size, full_seq_len, self.n_heads, self.head_dim)
        v = self.W_uv(c_kv).view(batch_size, full_seq_len, self.n_heads, self.head_dim)
        
        # Split Keys into Nope/Rope
        k_nope, k_pe = k.split([self.qk_nope_dim, self.qk_rope_dim], dim=-1)
        
        k_nope = k_nope.transpose(1, 2)
        k_pe = k_pe.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 5. Apply Dynamic RoPE to the positional parts
        # Fetch cos/sin based on the full sequence length (including cache)
        cos, sin = self.rope(v, full_seq_len)
        
        # Apply RoPE to current query window
        q_pe, _ = apply_rotary_pos_emb(q_pe, q_pe, cos[:, :, -seq_len:, :], sin[:, :, -seq_len:, :])
        # Apply RoPE to all keys in the buffer
        _, k_pe = apply_rotary_pos_emb(k_pe, k_pe, cos, sin)
        
        # Re-concatenate positional and non-positional components
        q_final = torch.cat([q_nope, q_pe], dim=-1)
        k_final = torch.cat([k_nope, k_pe], dim=-1)
        
        # 6. Attention Computation
        attn_scores = torch.matmul(q_final, k_final.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            if mask.dtype == torch.bool:
                attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
            else:
                attn_scores = attn_scores + mask
            
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        return self.W_o(context), present_c_kv