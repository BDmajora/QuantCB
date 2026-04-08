import torch
import torch.nn.functional as F
from typing import Optional, Tuple
# Ensure block.py, moe.py, attention.py, and ffn.py are in this directory
from block import quantcb_block_forward_stateless

def verify_block_stateless():
    print("--- Starting QuantCB Block Functional Verification ---")
    
    # 1. Hyperparameters
    d_model, n_heads, d_ff = 512, 8, 2048
    latent_dim, head_dim = 128, 64
    num_experts, top_k = 8, 2
    batch, seq = 2, 16
    qk_rope_dim = head_dim // 2

    # 2. Setup Mock Inputs
    x = torch.randn(batch, seq, d_model)
    mask = None 
    layer_past = None

    # 3. Initialize Mock Weights
    # --- Attention Weights ---
    attn_weights = {
        "wq_weight": torch.randn(n_heads * head_dim, d_model) * 0.02,
        "wdkv_weight": torch.randn(latent_dim, d_model) * 0.02,
        "ln_kv_weight": torch.ones(latent_dim),
        "ln_kv_bias": torch.zeros(latent_dim),
        "wuk_weight": torch.randn(n_heads * head_dim, latent_dim) * 0.02,
        "wuv_weight": torch.randn(n_heads * head_dim, latent_dim) * 0.02,
        "wo_weight": torch.randn(d_model, n_heads * head_dim) * 0.02,
        # Corrected RoPE freq shape for head_dim // 2
        "rope_inv_freq": 1.0 / (10000 ** (torch.arange(0, head_dim // 2, 2).float() / (head_dim // 2))),
    }

    # --- MoE Weights (SWIGLU 3-TENSOR STACK) ---
    # We no longer use a list of dicts. We use 3D tensors: [num_experts, out, in]
    moe_weights = {
        "router_weight": torch.randn(num_experts, d_model) * 0.02,
        "w1_weight": torch.randn(num_experts, d_ff, d_model) * 0.02, # Gate
        "w2_weight": torch.randn(num_experts, d_model, d_ff) * 0.02, # Down
        "w3_weight": torch.randn(num_experts, d_ff, d_model) * 0.02  # Up
    }

    # --- Top Level Block Weights ---
    # These keys MUST match the arguments in quantcb_block_forward_stateless
    weights = {
        "ln_1_weight": torch.ones(d_model),
        "ln_2_weight": torch.ones(d_model),
        "attn_weights": attn_weights,
        "moe_weights": moe_weights,
    }

    # 4. Run Verification
    try:
        with torch.no_grad():
            out_func, present_func, aux_func = quantcb_block_forward_stateless(
                x, 
                mask=mask, 
                layer_past=layer_past,
                n_heads=n_heads, 
                latent_dim=latent_dim, 
                head_dim=head_dim, 
                num_experts=num_experts, 
                top_k=top_k,
                **weights # This unpacks ln_1_weight, ln_2_weight, attn_weights, moe_weights
            )

        # 5. Checks
        print(f"[Test 1] Shape Verification")
        assert out_func.shape == (batch, seq, d_model), f"Output shape mismatch: {out_func.shape}"
        # MLA Cache shape is [batch, seq, latent_dim]
        assert present_func.shape == (batch, seq, latent_dim), f"Cache shape mismatch: {present_func.shape}"
        print(f"PASS: Output and Cache shapes are correct.")

        print(f"\n[Test 2] Numerical Sanity")
        assert not torch.isnan(out_func).any(), "NaN detected in output!"
        
        # Handle if aux_func is a tensor or scalar
        aux_val = aux_func.item() if isinstance(aux_func, torch.Tensor) else aux_func
        assert aux_val >= 0, "Auxiliary loss should be non-negative."
        print(f"PASS: No NaNs, Aux Loss is {aux_val:.4f}")

        print("\n--- ALL BLOCK TESTS PASSED ---")

    except Exception as e:
        print(f"\nFAILURE: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_block_stateless()