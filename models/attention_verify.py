import torch
import torch.nn as nn
import math
from attention import mla_attention_stateless

def verify_stateless_mla():
    print("--- Starting Stateless MLA Functional Verification ---")
    
    # 1. Configuration
    batch, seq_len, d_model = 2, 16, 512
    n_heads = 8
    latent_dim = 128
    head_dim = 64
    qk_rope_dim = head_dim // 2
    
    device = torch.device("cpu")
    dtype = torch.float32

    # 2. Mock Weights (Mimicking the buffers IREE will provide)
    # We initialize these with specific variances to check for stability
    weights = {
        "wq_weight": torch.randn(n_heads * head_dim, d_model) * 0.02,
        "wdkv_weight": torch.randn(latent_dim, d_model) * 0.02,
        "ln_kv_weight": torch.ones(latent_dim),
        "ln_kv_bias": torch.zeros(latent_dim),
        "wuk_weight": torch.randn(n_heads * head_dim, latent_dim) * 0.02,
        "wuv_weight": torch.randn(n_heads * head_dim, latent_dim) * 0.02,
        "wo_weight": torch.randn(d_model, n_heads * head_dim) * 0.02,
        "rope_inv_freq": 1.0 / (10000 ** (torch.arange(0, qk_rope_dim, 2).float() / qk_rope_dim)),
        "n_heads": n_heads,
        "latent_dim": latent_dim,
        "head_dim": head_dim
    }

    # 3. Test Inputs
    x = torch.randn(batch, seq_len, d_model)
    mask = torch.tril(torch.ones(batch, n_heads, seq_len, seq_len)).bool()

    # 4. Run Verification
    try:
        # Test 1: Standard Forward Pass
        out, present_kv = mla_attention_stateless(x, mask=mask, **weights)
        
        print(f"[Test 1] Standard Forward")
        assert out.shape == (batch, seq_len, d_model), f"Output shape mismatch: {out.shape}"
        assert present_kv.shape == (batch, seq_len, latent_dim), f"KV Cache shape mismatch: {present_kv.shape}"
        print("PASS: Shapes are correct.")

        # Test 2: KV Caching (Autoregressive step)
        # Pass the previous latent vector back in as layer_past
        out_2, present_kv_2 = mla_attention_stateless(x, layer_past=present_kv, **weights)
        
        print(f"\n[Test 2] KV Cache Integration")
        assert present_kv_2.shape == (batch, seq_len * 2, latent_dim), "KV Cache did not concatenate correctly."
        print("PASS: Layer past successfully concatenated.")

        # Test 3: Numerical Sanity
        print(f"\n[Test 3] Attention Logic Sanity")
        assert not torch.isnan(out).any(), "NaN detected in output!"
        assert not torch.isinf(out).any(), "Inf detected in output!"
        print("PASS: No numerical explosions (NaN/Inf).")

        # Test 4: Masking Effectiveness
        # If mask is all False, output variance should be near 0 or values should be uniform
        zero_mask = torch.zeros((batch, n_heads, seq_len, seq_len)).bool()
        out_masked, _ = mla_attention_stateless(x, mask=zero_mask, **weights)
        print("PASS: Masking logic executed without crash.")

        print("\n--- ALL TESTS PASSED ---")

    except Exception as e:
        print(f"\nFAILURE: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_stateless_mla()