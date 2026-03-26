import torch
# Updated import to find models/layers.py from the root
from models.layers import QuantCB_Block

def test_block_integrity():
    # Model Hyperparameters
    d_model = 512
    n_heads = 8
    d_ff = 2048
    latent_dim = 128
    head_dim = 64
    seq_len = 32
    batch_size = 2

    block = QuantCB_Block(d_model, n_heads, d_ff, latent_dim, head_dim)
    x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
    
    # Create a causal mask
    mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)

    # 1. Forward Pass
    output = block(x, mask=mask)
    
    # 2. Shape Verification
    assert output.shape == (batch_size, seq_len, d_model), f"Expected {(batch_size, seq_len, d_model)}, got {output.shape}"
    print("Shape Verification: PASSED")

    # 3. Numerical Stability (No NaNs)
    assert not torch.isnan(output).any(), "NaN detected in block output"
    print("Numerical Stability: PASSED")

    # 4. Gradient Flow
    loss = output.sum()
    loss.backward()
    assert x.grad is not None, "Gradients failed to propagate to input x"
    print("Gradient Flow: PASSED")

    # 5. Residual Integrity
    # If the block is zero-initialized or weights are small, output should resemble input
    # This is a soft check for the addition vs replacement logic
    assert not torch.equal(output, x), "Output is identical to input; block logic might be missing"
    print("Residual Integrity: PASSED")

if __name__ == "__main__":
    try:
        test_block_integrity()
        print("\nAll QuantCB Block tests passed successfully.")
    except Exception as e:
        print(f"\nTest failed: {e}")