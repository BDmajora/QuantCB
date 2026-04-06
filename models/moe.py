import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import QuantCB_FFN

class QuantCB_MoE(nn.Module):
    def __init__(self, d_model, d_ff, num_experts=8, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # The Router: Decides which expert gets which token
        self.router = nn.Linear(d_model, num_experts, bias=False)
        
        # The Experts: A collection of standard FFNs
        self.experts = nn.ModuleList([
            QuantCB_FFN(d_model, d_ff) for _ in range(num_experts)
        ])

    def forward(self, x):
        batch, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)  # (Total_Tokens, d_model)
        
        # 1. Get Router Scores
        router_logits = self.router(x_flat)
        weights = F.softmax(router_logits, dim=-1)
        
        # 2. Identify Top-K Experts
        # We still use topk to find which experts are winners, but we 
        # won't use these to 'slice' the tensor anymore.
        _, top_k_indices = torch.topk(weights, self.top_k, dim=-1)
        
        # 3. Create a Static Weight Mask (The "IREE Fix")
        # We create a mask of zeros and 'scatter' 1.0s into the top-k positions.
        # This keeps the shape [Total_Tokens, num_experts] perfectly static.
        mask = torch.zeros_like(weights)
        mask.scatter_(1, top_k_indices, 1.0)
        
        # Apply the mask to our weights and re-normalize.
        # Now, 'static_weights' is zero for any expert not in the top-k.
        masked_weights = weights * mask
        static_weights = masked_weights / (masked_weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        # 4. Static Aggregation
        # Instead of finding indices and slicing, we run the batch through 
        # the experts and use the weights to "gate" the results.
        out = torch.zeros_like(x_flat)
        
        for i in range(self.num_experts):
            # We process the full x_flat. While this seems redundant, 
            # IREE/SPIR-V can optimize this much better than dynamic indexing.
            expert_out = self.experts[i](x_flat)
            
            # Only tokens assigned to expert 'i' will have a non-zero weight here.
            # This is essentially a "soft scatter" that the compiler loves.
            expert_contribution = expert_out * static_weights[:, i].unsqueeze(-1)
            out = out + expert_contribution
                
        return out.view(batch, seq_len, d_model)