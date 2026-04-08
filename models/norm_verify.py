import torch
import torch.nn as nn

def rms_norm_stateless(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    r"""
    Pure functional RMSNorm.
    Formula: $$y = \frac{x}{\sqrt{\text{mean}(x^2) + \epsilon}} * w$$
    """
    norm_x = torch.mean(x.pow(2), dim=-1, keepdim=True)
    x_normed = x * torch.rsqrt(norm_x + eps)
    return weight * x_normed

def verify_rms_norm_stateless():
    # Use a raw string here as well if you include the LaTeX formula in the print
    print("--- Starting RMSNorm Functional Verification ---")
    
    # 1. Setup Mock Data
    batch, seq, d_model = 2, 4, 16
    x = torch.randn(batch, seq, d_model) * 5.0  # High variance input
    weight = torch.ones(d_model) * 0.5           # Scaling weight
    eps = 1e-6

    # 2. Run Functional Pass
    out = rms_norm_stateless(x, weight, eps)

    # 3. Verification Logic
    try:
        # [Test 1] Shape Check
        assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"
        print(f"PASS: Output shape matches input {out.shape}")

        # [Test 2] Numerical Sanity (NaN/Inf)
        assert not torch.isnan(out).any(), "NaN detected in output!"
        assert not torch.isinf(out).any(), "Inf detected in output!"
        print("PASS: No NaNs or Infs detected.")

        # [Test 3] RMS Scaling Check
        # Before the weight is applied, the RMS should be ~1.0
        # Since we applied weight=0.5, the RMS of the output should be ~0.5
        rms_val = torch.sqrt(torch.mean(out.pow(2), dim=-1))
        mean_rms = rms_val.mean().item()
        
        # Check if mean_rms is close to our weight (0.5)
        assert abs(mean_rms - 0.5) < 1e-3, f"RMS scaling failed: expected ~0.5, got {mean_rms:.4f}"
        print(f"PASS: RMS scaling is correct ({mean_rms:.4f})")

        # [Test 4] Comparison with Manual Reference
        # Ensure the math hasn't diverged from the standard iterative approach
        expected = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight
        torch.testing.assert_close(out, expected)
        print("PASS: Functional output matches manual reference calculation.")

        print("\n--- ALL RMSNORM TESTS PASSED ---")

    except Exception as e:
        print(f"\nFAILURE: {str(e)}")

if __name__ == "__main__":
    verify_rms_norm_stateless()