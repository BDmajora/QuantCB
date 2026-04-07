import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import QuantCB_FFN

class QuantCB_MoE(nn.Module):
    def __init__(self, d_model, d_ff, num_experts=8, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList([
            QuantCB_FFN(d_model, d_ff) for _ in range(num_experts)
        ])
        
        # Registration of constants as buffers prevents "Lifted Tensor" warnings
        self.register_buffer("eps", torch.tensor(1e-8))
        self.register_buffer("zero_loss", torch.tensor(0.0))

    def forward(self, x):
        batch, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)  # (Total_Tokens, d_model)
        
        # 1. Get Router Scores
        router_logits = self.router(x_flat)
        weights = F.softmax(router_logits, dim=-1)
        
        # 2. Identify Top-K Experts
        _, topk_indices = torch.topk(weights, self.top_k, dim=-1)
        
        # 3. Create a Static Mask (Functional)
        mask = torch.zeros_like(weights)
        mask = mask.scatter(1, topk_indices, 1.0)
        
        # 4. Apply Mask and Re-normalize
        masked_weights = weights * mask
        denom = masked_weights.sum(dim=-1, keepdim=True) + self.eps
        static_weights = masked_weights / denom
        
        # 5. Fused Expert Aggregation (The IREE Optimization)
        # We stack expert outputs into a single [Tokens, Experts, D_Model] tensor.
        # Then use Einstein Summation to multiply by weights and sum across experts.
        expert_outs = torch.stack([expert(x_flat) for expert in self.experts], dim=1)
        
        # 'be' = (Tokens, Experts), 'bed' = (Tokens, Experts, D_Model) -> 'bd' = (Tokens, D_Model)
        out = torch.einsum('be,bed->bd', static_weights, expert_outs)
        
        # 6. Load Balancing Loss
        if self.training:
            importance = weights.mean(dim=0)
            load = mask.mean(dim=0)
            aux_loss = (importance * load).sum() * self.num_experts
        else:
            # Return the pre-registered zero buffer
            aux_loss = self.zero_loss
                
        return out.view(batch, seq_len, d_model), aux_loss