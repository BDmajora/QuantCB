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
        
        # Query projection (Standard)
        self.W_q = nn.Linear(d_model, n_heads * head_dim, bias=False)
        
        # KV Compression (The MLA Secret Sauce)
        # Down-project input to a smaller latent dimension
        self.W_dkv = nn.Linear(d_model, latent_dim, bias=False)
        self.ln_kv = nn.LayerNorm(latent_dim)
        
        # Up-project latent vector to full multi-head keys and values
        self.W_uk = nn.Linear(latent_dim, n_heads * head_dim, bias=False)
        self.W_uv = nn.Linear(latent_dim, n_heads * head_dim, bias=False)
        
        # Output projection
        self.W_o = nn.Linear(n_heads * head_dim, d_model, bias=False)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # 1. Generate Queries
        q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        q = q.transpose(1, 2) # (batch, n_heads, seq_len, head_dim)
        
        # 2. Compress to Latent Vector (This is what you would cache in inference)
        c_kv = self.ln_kv(self.W_dkv(x))
        
        # 3. Expand Latent to Keys and Values
        k = self.W_uk(c_kv).view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = self.W_uv(c_kv).view(batch_size, seq_len, self.n_heads, self.head_dim)
        
        k = k.transpose(1, 2) # (batch, n_heads, seq_len, head_dim)
        v = v.transpose(1, 2) # (batch, n_heads, seq_len, head_dim)
        
        # 4. Standard Scaled Dot-Product Attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
            
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # 5. Context Calculation and Projection
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        return self.W_o(context)