import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MLA_Attention(nn.Module):
    def __init__(self, d_model, n_heads, latent_dim=128, head_dim=64):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.latent_dim = latent_dim
        
        self.W_q = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.W_dkv = nn.Linear(d_model, latent_dim, bias=False)
        self.ln_kv = nn.LayerNorm(latent_dim)
        
        self.W_uk = nn.Linear(latent_dim, n_heads * head_dim, bias=False)
        self.W_uv = nn.Linear(latent_dim, n_heads * head_dim, bias=False)
        self.W_o = nn.Linear(n_heads * head_dim, d_model, bias=False)

        # Precompute the inverse frequencies for RoPE
        # We register this as a buffer so it automatically moves to CPU/CUDA with the model
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)

    def apply_rope(self, x, seq_len, offset=0):
        # x shape: (batch, n_heads, seq_len, head_dim)
        t = torch.arange(offset, offset + seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim=-1)
        
        # Reshape for broadcasting across batch and heads
        freqs = freqs.unsqueeze(0).unsqueeze(0)
        
        # Rotate half the dimensions
        d = x.shape[-1] // 2
        x_half1, x_half2 = x[..., :d], x[..., d:]
        rotated_x = torch.cat((-x_half2, x_half1), dim=-1)
        
        # Apply sine and cosine to bake the position into the tensor
        return (x * freqs.cos()) + (rotated_x * freqs.sin())

    def forward(self, x, mask=None, layer_past=None):
        batch_size, seq_len, _ = x.shape
        
        # 1. Generate Queries
        q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        q = q.transpose(1, 2) 
        
        # 2. Compress current tokens to Latent Vector
        c_kv = self.ln_kv(self.W_dkv(x)) 
        
        # 3. Cache Update
        if layer_past is not None:
            c_kv = torch.cat([layer_past, c_kv], dim=1)
            
        present_c_kv = c_kv 
        
        # 4. Expand Latent Vector to Keys and Values
        full_seq_len = c_kv.size(1)
        k = self.W_uk(c_kv).view(batch_size, full_seq_len, self.n_heads, self.head_dim)
        v = self.W_uv(c_kv).view(batch_size, full_seq_len, self.n_heads, self.head_dim)
        
        k = k.transpose(1, 2) 
        v = v.transpose(1, 2) 
        
        # 5. Apply Rotary Position Embeddings
        # Calculate offset in case we are in the middle of generation using a KV cache
        offset = full_seq_len - seq_len
        q = self.apply_rope(q, seq_len, offset)
        k = self.apply_rope(k, full_seq_len, offset=0)
        
        # 6. Attention Computation
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            if mask.dtype == torch.bool:
                attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
            else:
                attn_scores = attn_scores + mask
            
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        return self.W_o(context), present_c_kv