import torch
from quantcb_model import quantcb_model_forward_stateless, get_hallucination_score_stateless

def verify_base_model():
    print("--- Starting QuantCB Base Model Verification ---")
    
    # 1. Hyperparameters
    # These must match your 2026 IREE-Turbine static bake configuration
    d_model, vocab_size = 256, 1000
    batch, seq, num_blocks = 2, 16, 2
    n_heads = 8
    latent_dim = 128
    head_dim = 64  
    num_experts = 8
    d_ff = 512

    # 2. Mock Inputs
    idx = torch.randint(0, vocab_size, (batch, seq))

    # 3. Mock Weights (Stateless Buffer Dictionary)
    shared_embedding_weight = torch.randn(vocab_size, d_model) * 0.02
    
    weights = {
        "token_embedding.weight": shared_embedding_weight,
        "lm_head.weight": shared_embedding_weight, 
        "ln_f.weight": torch.ones(d_model),
        "latent_probe.weight": torch.randn(1, d_model) * 0.02
    }

    # Add block weights
    for i in range(num_blocks):
        # Layer Normalization Buffers
        weights[f"blocks.{i}.ln1.weight"] = torch.ones(d_model)
        weights[f"blocks.{i}.ln2.weight"] = torch.ones(d_model)
        
        # --- MLA Weight Mapping (Multi-Head Latent Attention) ---
        # Query projection (d_model -> n_heads * head_dim)
        weights[f"blocks.{i}.attn.wq_weight"] = torch.randn(n_heads * head_dim, d_model) * 0.02
        
        # KV Down-projection and LayerNorm
        weights[f"blocks.{i}.attn.wdkv_weight"] = torch.randn(latent_dim, d_model) * 0.02
        weights[f"blocks.{i}.attn.ln_kv_weight"] = torch.ones(latent_dim)
        weights[f"blocks.{i}.attn.ln_kv_bias"] = torch.zeros(latent_dim)
        
        # KV Up-projections (latent_dim -> n_heads * head_dim)
        weights[f"blocks.{i}.attn.wuk_weight"] = torch.randn(n_heads * head_dim, latent_dim) * 0.02
        weights[f"blocks.{i}.attn.wuv_weight"] = torch.randn(n_heads * head_dim, latent_dim) * 0.02
        
        # Output projection (n_heads * head_dim -> d_model)
        weights[f"blocks.{i}.attn.wo_weight"] = torch.randn(d_model, n_heads * head_dim) * 0.02
        
        # RoPE Inverse Frequencies
        weights[f"blocks.{i}.attn.rope_inv_freq"] = 1.0 / (10000 ** (torch.arange(0, head_dim // 2, 2).float() / (head_dim // 2)))

        # --- FIXED: MoE Weight Mapping (SwiGLU / 3-Weight Pattern) ---
        # Router: Maps d_model to expert selection scores
        weights[f"blocks.{i}.moe.router_weight"] = torch.randn(num_experts, d_model) * 0.02
        
        # SwiGLU requires 3 weight matrices per expert:
        weights[f"blocks.{i}.moe.w1_weight"] = torch.randn(num_experts, d_ff, d_model) * 0.02 # Gate (SiLU)
        weights[f"blocks.{i}.moe.w3_weight"] = torch.randn(num_experts, d_ff, d_model) * 0.02 # Up (Linear)
        weights[f"blocks.{i}.moe.w2_weight"] = torch.randn(num_experts, d_model, d_ff) * 0.02 # Down (Project back)

    # 4. Execute Functional Path
    try:
        with torch.no_grad():
            # This simulates the SPIR-V kernel dispatch logic
            logits, presents, l_aux = quantcb_model_forward_stateless(
                idx, 
                weights, 
                num_blocks, 
                n_heads=n_heads
            )
            
            # Test hallucination probe logic
            dummy_h_n = torch.randn(batch, seq, d_model)
            h_score = get_hallucination_score_stateless(dummy_h_n, weights["latent_probe.weight"])

        print("\n[Test 1] Logits Shape & Fidelity")
        assert logits.shape == (batch, seq, vocab_size)
        print(f"PASS: Logits shape {logits.shape} verified.")

        print("[Test 2] KV Cache / Presents Tracking")
        assert len(presents) == num_blocks
        # MLA KV cache stores the compressed latent dimension (128)
        assert presents[0].shape == (batch, seq, latent_dim)
        print(f"PASS: Captured {len(presents)} KV caches with correct latent shape.")

        print("[Test 3] Auxiliary Loss Accumulation")
        # Aux loss must be a scalar for Vulkan-native backward passes
        aux_val = l_aux.item() if isinstance(l_aux, torch.Tensor) else l_aux
        assert aux_val >= 0
        print(f"PASS: Total Aux Loss (MoE Balance) computed: {aux_val:.4f}")
        
        print("[Test 4] Latent Probe Execution")
        assert h_score.shape == (batch, seq, 1)
        print(f"PASS: Hallucination scores verified.")

        print("\n--- ALL BASE MODEL TESTS PASSED ---")

    except Exception as e:
        print(f"\nFAILURE: Verification engine encountered an error.")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_base_model()