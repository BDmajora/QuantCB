import torch
# Root-relative import for the new directory structure
from models.layers import QuantCB_FFN

def test_ffn_independence():
    print("--- Running FFN Position-wise Test ---")
    d_model, d_ff = 256, 1024
    ffn = QuantCB_FFN(d_model, d_ff)
    ffn.eval()

    # Create two sequences: identical except for the LAST token
    x1 = torch.randn(1, 4, d_model)
    x2 = x1.clone()
    x2[0, -1, :] = torch.randn(d_model)

    with torch.no_grad():
        out1 = ffn(x1)
        out2 = ffn(x2)

    # In FFN, the change at index 3 MUST NOT affect indices 0-2
    diff = torch.abs(out1[0, :3, :] - out2[0, :3, :]).max().item()
    
    print(f"Input shape:  {x1.shape}")
    print(f"Output shape: {out1.shape}")
    print(f"Max deviation in other tokens: {diff:.10f}")

    assert out1.shape == x1.shape
    assert diff == 0, "FFN Error: Information leaked between positions!"
    print("SUCCESS: FFN is strictly position-wise.")

if __name__ == "__main__":
    test_ffn_independence()