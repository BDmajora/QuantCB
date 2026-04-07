import torch
import torch.nn as nn
import torch.nn.functional as F

class MTPModule(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        
        # 1. Projections (Explicitly registered)
        self.proj_h = nn.Linear(d_model, d_model, bias=False)
        self.proj_emb = nn.Linear(d_model, d_model, bias=False)
        self.ln_fusion = nn.LayerNorm(d_model)
        
        # 2. Manual Transformer Block (Replacement for TransformerEncoderLayer)
        # Standard built-in layers often cause "lifted tensor" issues in Turbine/IREE.
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln_1 = nn.LayerNorm(d_model)
        
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.GELU(),
            nn.Linear(d_ff, d_model, bias=False)
        )
        self.ln_2 = nn.LayerNorm(d_model)
        
        self._init_mtp_weights()

    def _init_mtp_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, h_base, x_embed, mask=None):
        # Fusion Step
        # Ensure we don't create new tensors with grad manually here.
        fused = (self.proj_h(h_base) + self.proj_emb(x_embed)) * 0.5
        x = self.ln_fusion(fused)
        
        # Self-Attention Block (Explicit Residuals)
        # Using functional mask passing to avoid internal buffer issues.
        attn_out, _ = self.attn(x, x, x, attn_mask=mask, need_weights=False)
        x = self.ln_1(x + attn_out)
        
        # MLP Block
        x = self.ln_2(x + self.mlp(x))
        
        return x