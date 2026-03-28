import torch
import torch.nn as nn
from models.layers import QuantCB_Block, RMSNorm 
from models.mtp_module import MTPModule

class QuantCB_Model(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=8, d_ff=1024, n_layers=6, 
                 latent_dim=128, head_dim=64, num_experts=8, top_k=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        
        # 1. Base Components
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Consistent initialization for tied weights
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        
        # 2. Transformer Stack
        self.blocks = nn.ModuleList([
            QuantCB_Block(
                d_model=d_model, 
                n_heads=n_heads, 
                d_ff=d_ff, 
                latent_dim=latent_dim, 
                head_dim=head_dim, 
                num_experts=num_experts, 
                top_k=top_k, 
                dropout=dropout
            ) 
            for _ in range(n_layers)
        ])
        
        # 3. Final Norm and Tied Head
        self.ln_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        # 4. Integrated MTP Module
        self.mtp = MTPModule(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            embedding=self.token_embedding,
            head=self.lm_head
        )

        # 5. Latent Probing Head
        # Projects d_model to a hallucination/drift score for the Ouro_Engine
        self.latent_probe = nn.Linear(d_model, 1, bias=False)
        nn.init.normal_(self.latent_probe.weight, std=0.02)

    def get_hallucination_score(self, h_n):
        """
        Inspects the latent state to detect logic drift.
        Used by the engine to decide on early exits or re-loops.
        """
        return torch.sigmoid(self.latent_probe(h_n))