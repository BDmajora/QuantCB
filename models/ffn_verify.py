import torch
import torch.nn as nn
import torch.nn.functional as F

# Assuming your function is saved in ffn.py
# try/except block added for easy execution from any directory
try:
    from .ffn import swiglu_ffn_stateless
except (ImportError, ValueError):
    # Fallback definition for standalone testing if import fails
    def swiglu_ffn_stateless(x: torch.Tensor, w1_weight: torch.Tensor, w2_weight: torch.Tensor, w3_weight: torch.Tensor) -> torch.Tensor:
        gate = F.linear(x, w1_weight)
        up = F.linear(x, w3_weight)
        hidden = F.silu(gate) * up
        return F.linear(hidden, w2_weight)

# --- Stateful Reference Model for Parity Testing ---
class ReferenceSwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.w1(x)
        up = self.w3(x)
        hidden = F.silu(gate) * up
        return self.w2(hidden)

def verify_swiglu_stateless():
    print("--- Starting SwiGLU FFN Functional Verification ---")
    
    # 1. Hyperparameters
    # Note: In LLaMA/DeepSeek, d_ff is usually computed as an int multiple of a fraction of d_model.
    # For testing, we just use standard arbitrary sizes.
    batch, seq_len = 2, 16
    d_model = 512
    d_ff = 1376  # Typically somewhat smaller than 4x d_model in SwiGLU (e.g., 8/3 * d_model)

    device = torch.device("cpu")

    # 2. Initialize Reference Model (Stateful)
    reference_model = ReferenceSwiGLUFFN(d_model, d_ff)
    reference_model.eval()

    # 3. Setup Inputs
    x = torch.randn(batch, seq_len, d_model)

    # 4. Extract Weights for the Stateless Function
    weights = {
        "w1_weight": reference_model.w1.weight,
        "w2_weight": reference_model.w2.weight,
        "w3_weight": reference_model.w3.weight
    }

    # 5. Run Verification
    try:
        with torch.no_grad():
            # Run original stateful pass
            out_orig = reference_model(x)
            
            # Run new stateless functional pass
            out_func = swiglu_ffn_stateless(x, **weights)

        # [Test 1] Shape Verification
        print(f"\n[Test 1] Shape Verification")
        assert out_func.shape == (batch, seq_len, d_model), f"Output shape mismatch: {out_func.shape}"
        print(f"PASS: Output shape is correct: {out_func.shape}")

        # [Test 2] Numerical Sanity
        print(f"\n[Test 2] Numerical Sanity")
        assert not torch.isnan(out_func).any(), "NaN detected in output!"
        assert not torch.isinf(out_func).any(), "Inf detected in output!"
        print(f"PASS: No NaNs or Infs detected.")

        # [Test 3] Mathematical Parity
        print(f"\n[Test 3] Mathematical Parity")
        is_correct = torch.allclose(out_orig, out_func, atol=1e-5)
        
        if is_correct:
            print("PASS:The stateless function is mathematically identical to the stateful module.")
        else:
            diff = (out_orig - out_func).abs().max().item()
            print(f"FAIL:Outputs diverge. Maximum difference: {diff}")

        print("\n--- ALL FFN TESTS PASSED ---")

    except Exception as e:
        print(f"\nFAILURE: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_swiglu_stateless()