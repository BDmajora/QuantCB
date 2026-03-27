import torch
import torch.nn as nn
import torch.nn.functional as F

class MTPModule(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, embedding, head):
        super().__init__()
        self.embedding = embedding 
        self.head = head           
        
        self.fusion = nn.Linear(2 * d_model, d_model)
        self.ln_fusion = nn.LayerNorm(d_model)
        
        # Proper Transformer layer for MTP logic
        self.layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=d_ff, 
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

    def forward(self, h_base, targets):
        x_embed = self.embedding(targets)
        fused = torch.cat([h_base, x_embed], dim=-1)
        x = self.fusion(fused)
        x = self.ln_fusion(x)
        x = self.layer(x)
        return self.head(x)

class QuantCBModel(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, num_layers):
        super().__init__()
        # 1. Base Model Components
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.base_blocks = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, 
                activation="gelu", batch_first=True, norm_first=True
            ),
            num_layers=num_layers
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight Tying: The head uses the embedding weights
        self.lm_head.weight = self.embedding.weight
        
        # 2. The MTP Module (Passing the shared weights)
        self.mtp = MTPModule(
            d_model=d_model, 
            n_heads=n_heads, 
            d_ff=d_ff, 
            embedding=self.embedding, 
            head=self.lm_head
        )
        
        # MTP Loss Weight (DeepSeek-V3 uses a coefficient to balance the loss)
        self.mtp_loss_weight = 0.3

    def forward(self, input_ids):
        """
        Expects input_ids of shape [batch_size, sequence_length]
        """
        # --- PREPARATION ---
        # For a sequence of length L, base input is L-1, base target is L-1
        x = input_ids[:, :-1]          # Inputs (t)
        base_targets = input_ids[:, 1:] # Targets (t+1)
        
        # --- BASE MODEL FORWARD ---
        embeds = self.embedding(x)
        
        # Causal mask for the base model
        seq_len = x.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        
        h_base = self.base_blocks(embeds, mask=causal_mask, is_causal=True)
        base_logits = self.lm_head(h_base)
        
        # 1. Calculate Base Loss (Predicting t+1)
        loss_fct = nn.CrossEntropyLoss()
        base_loss = loss_fct(base_logits.reshape(-1, base_logits.size(-1)), base_targets.reshape(-1))

        # --- MTP DATA REALIGNMENT ---
        # To predict t+2, we must drop the last token from our current states 
        # because we don't have the ground truth for it.
        
        # h_base slice: Everything except the last step
        h_base_mtp = h_base[:, :-1, :] 
        
        # mtp_inputs: The actual tokens at t+1 (to feed into the MTP embedding)
        mtp_inputs = base_targets[:, :-1]
        
        # mtp_targets: The actual tokens at t+2 (the ground truth for MTP loss)
        mtp_targets = base_targets[:, 1:]

        # --- MTP FORWARD ---
        mtp_logits = self.mtp(h_base_mtp, mtp_inputs)
        
        # 2. Calculate MTP Loss (Predicting t+2)
        mtp_loss = loss_fct(mtp_logits.reshape(-1, mtp_logits.size(-1)), mtp_targets.reshape(-1))

        # --- TOTAL LOSS ---
        total_loss = base_loss + (self.mtp_loss_weight * mtp_loss)
        
        return {
            "loss": total_loss,
            "base_loss": base_loss,
            "mtp_loss": mtp_loss,
            "base_logits": base_logits
        }