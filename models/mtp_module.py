import torch
import torch.nn as nn
import torch.nn.functional as F

class MTPModule(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, embedding, head):
        super().__init__()
        # These are shared with the base model - DO NOT re-initialize these!
        self.embedding = embedding 
        self.head = head           
        
        # 1. Projections for Fusion
        self.proj_h = nn.Linear(d_model, d_model, bias=False)
        self.proj_emb = nn.Linear(d_model, d_model, bias=False)
        self.ln_fusion = nn.LayerNorm(d_model)
        
        # 2. Transformer Layer (The Mixer)
        self.layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=d_ff, 
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        
        # 3. Targeted Initialization
        self._init_mtp_weights(self.proj_h)
        self._init_mtp_weights(self.proj_emb)
        self._init_mtp_weights(self.ln_fusion)
        self._init_mtp_weights(self.layer)

    def _init_mtp_weights(self, module):
        """DeepSeek-style: Initialize MTP specific layers to be very 'quiet' at start."""
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, h_base, targets):
        """
        h_base: Hidden states from base model at t
        targets: Actual tokens at t+1 (the hint)
        Predicts: Tokens at t+2
        """
        # 1. Get embeddings for the 'hint' tokens (t+1)
        x_embed = self.embedding(targets)
        
        # 2. Mix: DeepSeek-V3 style additive fusion
        fused = (self.proj_h(h_base) + self.proj_emb(x_embed)) * 0.5
        x = self.ln_fusion(fused)
        
        # 3. Generate Causal Mask
        # We use the built-in utility to ensure the mask format matches what 
        # the C++ dispatcher expects for the 'is_causal' hint.
        sz = x.size(1)
        mask = nn.Transformer.generate_square_subsequent_mask(sz, device=x.device)
        
        # 4. Process through MTP-specific Transformer block
        # Providing both the 'src_mask' and 'is_causal=True' satisfies the 
        # validation check and allows for optimized attention kernels.
        x_mtp = self.layer(x, src_mask=mask, is_causal=True)
        
        # 5. Predict t+2 using the shared head
        logits = self.head(x_mtp)
        
        return logits, x_mtp