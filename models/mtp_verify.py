import torch
import torch.nn.functional as F
from mtp import mtp_forward_stateless

def verify_mtp_stateless():
    print("--- Starting MTP Functional Verification (Stateless) ---")
    
    # 1. Hyperparameters
    d_model, n_heads, d_ff = 512, 8, 2048
    vocab_size = 1000
    batch, seq = 2, 16
    device = torch.device("cpu")

    # 2. Setup Mock Inputs
    # h_base represents the hidden states from the main transformer trunk
    h_base = torch.randn(batch, seq, d_model)
    # targets are the tokens for the next-token prediction heads
    targets = torch.randint(0, vocab_size, (batch, seq))

    # 3. Initialize Mock Weights (Simulating IREE buffers)
    # We use small standard deviations to keep the math stable
    weights = {
        "embedding_weight": torch.randn(vocab_size, d_model) * 0.02,
        "head_weight": torch.randn(vocab_size, d_model) * 0.02,
        "proj_h_weight": torch.randn(d_model, d_model) * 0.02,
        "proj_emb_weight": torch.randn(d_model, d_model) * 0.02,
        "ln_fusion_weight": torch.ones(d_model),
        "ln_fusion_bias": torch.zeros(d_model),
        "norm1_weight": torch.ones(d_model),
        "norm1_bias": torch.zeros(d_model),
        "attn_qkv_weight": torch.randn(3 * d_model, d_model) * 0.02,
        "attn_qkv_bias": torch.zeros(3 * d_model),
        "attn_out_weight": torch.randn(d_model, d_model) * 0.02,
        "attn_out_bias": torch.zeros(d_model),
        "norm2_weight": torch.ones(d_model),
        "norm2_bias": torch.zeros(d_model),
        "mlp_fc1_weight": torch.randn(d_ff, d_model) * 0.02,
        "mlp_fc1_bias": torch.zeros(d_ff),
        "mlp_fc2_weight": torch.randn(d_model, d_ff) * 0.02,
        "mlp_fc2_bias": torch.zeros(d_model),
    }

    # 4. Run Functional Forward Pass
    try:
        with torch.no_grad():
            logits, loss = mtp_forward_stateless(
                h_base, 
                targets, 
                **weights, 
                n_heads=n_heads
            )

        # 5. Verification Checks
        print(f"[Test 1] Shape Verification")
        # Expected shape: (batch, seq, vocab_size)
        assert logits.shape == (batch, seq, vocab_size), f"Logits shape mismatch: {logits.shape}"
        print(f"PASS: Logits shape is {logits.shape}")

        print(f"\n[Test 2] Numerical Sanity")
        assert not torch.isnan(logits).any(), "NaN detected in logits!"
        assert not torch.isinf(logits).any(), "Inf detected in logits!"
        
        # Handle cases where loss might be a multi-element tensor
        mean_loss = loss.mean().item()
        assert mean_loss > 0, f"Loss should be positive, got {mean_loss}"
        
        print(f"PASS: Loss value is {mean_loss:.4f} (No NaNs/Infs)")

        print("\n--- ALL MTP TESTS PASSED ---")

    except Exception as e:
        print(f"\nFAILURE: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_mtp_stateless()