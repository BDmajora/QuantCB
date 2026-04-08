import torch
import torch.nn.functional as F
from typing import Tuple

# JUST IMPORT IT!
from ffn import swiglu_ffn_stateless 

def quantcb_moe_stateless(
    x: torch.Tensor,
    # --- MoE Weights ---
    router_weight: torch.Tensor,
    w1_weight: torch.Tensor, # Gate
    w2_weight: torch.Tensor, # Down
    w3_weight: torch.Tensor, # Up (NEW)
    # --- Config ---
    num_experts: int = 8,
    top_k: int = 2
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pure functional Mixture of Experts using SwiGLU."""
    batch, seq_len, d_model = x.shape
    x_flat = x.view(-1, d_model)
    
    # 1. Router Pass
    router_logits = F.linear(x_flat, router_weight)
    weights = F.softmax(router_logits, dim=-1)
    
    # 2. Select Top-K
    top_k_weights, top_k_indices = torch.topk(weights, top_k, dim=-1)
    
    # 3. Static Weight Masking for Load Balancing
    expert_mask = F.one_hot(top_k_indices, num_classes=num_experts).float()
    combined_weights = (top_k_weights.unsqueeze(-1) * expert_mask).sum(dim=1)
    
    # 4. Load Balancing Loss
    mean_probs = weights.mean(dim=0)
    density_probs = combined_weights.mean(dim=0)
    l_aux = num_experts * torch.sum(mean_probs * density_probs)
    
    # 5. Static Aggregation Loop
    final_output = torch.zeros_like(x_flat)
    
    for i in range(num_experts):
        # Call your actual FFN from ffn.py
        expert_out = swiglu_ffn_stateless(
            x_flat, 
            w1_weight[i], 
            w2_weight[i],
            w3_weight[i]  # Pass the 3rd weight
        )
        
        # Apply the router's gating weight for this expert
        gate_i = combined_weights[:, i].unsqueeze(-1)
        final_output = final_output + (expert_out * gate_i)
            
    return final_output.view(batch, seq_len, d_model), l_aux