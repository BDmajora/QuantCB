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
        x_flat = x.view(-1, d_model) # (Batch * Seq, d_model)
        
        # 1. Get Router Scores
        # logits: (Total_Tokens, num_experts)
        router_logits = self.router(x_flat)
        weights = F.softmax(router_logits, dim=-1)
        
        # 2. Select Top-K Experts
        top_k_weights, top_k_indices = torch.topk(weights, self.top_k, dim=-1)
        
        # Normalize weights so they sum to 1
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        # 3. Dispatch and Aggregate
        # For simplicity in this implementation, we iterate through tokens.
        # (In a production 'Flash' version, we would use grouped gemms)
        out = torch.zeros_like(x_flat)
        
        for i in range(self.num_experts):
            # Find which tokens in the batch were assigned to expert 'i'
            token_indices, expert_rank = torch.where(top_k_indices == i)
            
            if token_indices.numel() > 0:
                # Get the tokens assigned to this expert
                expert_input = x_flat[token_indices]
                # Run the expert
                expert_out = self.experts[i](expert_input)
                # Multiply by the router weight and add to output
                out[token_indices] += expert_out * top_k_weights[token_indices, expert_rank].unsqueeze(-1)
                
        return out.view(batch, seq_len, d_model)