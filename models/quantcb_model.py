import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import QuantCB_Block, RMSNorm 
from models.mtp_module import MTPModule

class QuantCB_Model(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=8, d_ff=1024, n_layers=6, 
                 latent_dim=128, head_dim=64, num_experts=8, top_k=2, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        
        # 1. Base Components
        # Input Embedding (Weight will be tied to the Output Head)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        
        # 2. Transformer Stack
        self.blocks = nn.ModuleList([
            QuantCB_Block(
                d_model=d_model, n_heads=n_heads, d_ff=d_ff, 
                latent_dim=latent_dim, head_dim=head_dim, 
                num_experts=num_experts, top_k=top_k, dropout=dropout
            ) for _ in range(n_layers)
        ])
        
        # 3. Normalization & Probing
        self.ln_f = RMSNorm(d_model)
        self.latent_probe = nn.Linear(d_model, 1, bias=False)

    def forward(self, x, target=None, mask=None):
        """
        Returns: (logits, loss, hidden_states)
        """
        h = self.token_embedding(x)
        
        # Initialize aux loss tied to the activation tensor device/dtype
        total_aux_loss = h.new_zeros((), dtype=torch.float32)
        
        for block in self.blocks:
            # Catching (hidden, aux_loss, _)
            h, l_aux, _ = block(h, mask=mask)
            total_aux_loss = total_aux_loss + l_aux
            
        h = self.ln_f(h)
        
        # Tied Output Head
        logits = F.linear(h, self.token_embedding.weight)
        
        loss = h.new_zeros((), dtype=torch.float32)
        if target is not None:
            # Split CE into log_softmax + nll_loss for SPIR-V stability
            log_probs = F.log_softmax(logits.view(-1, self.vocab_size), dim=-1)
            main_loss = F.nll_loss(log_probs, target.view(-1), ignore_index=-1)
            loss = main_loss + total_aux_loss
            
        return logits, loss, h