import torch
import torch.nn as nn
import torch.nn.functional as F

class MTPModule(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, embedding, head):
        super().__init__()
        self.embedding = embedding 
        self.head = head           
        
        # 1. Simplified Fusion
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
        
        # 3. FIX: Targeted Initialization. 
        # Apply ONLY to MTP-specific layers to prevent overwriting the shared LM Head and Embeddings.
        self.proj_h.apply(self._init_weights)
        self.proj_emb.apply(self._init_weights)
        self.layer.apply(self._init_weights)
        self.ln_fusion.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.01)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, h_base, targets):
        x_embed = self.embedding(targets)
        
        fused = self.proj_h(h_base) + self.proj_emb(x_embed)
        x = self.ln_fusion(fused)
        
        # FIX: Generate a Causal Mask to prevent the MTP layer from looking into the future
        seq_length = x.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_length, device=x.device
        )
        
        # Apply the mask. is_causal=True optimizes execution on PyTorch 2.0+
        x_mtp = self.layer(x, src_mask=causal_mask, is_causal=True)
        
        logits = self.head(x_mtp)
        
        return logits, x_mtp