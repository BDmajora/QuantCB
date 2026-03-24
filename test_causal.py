import torch
from models.attention import QuantCB_Attention

def test_causal_integrity():
    d_model, n_heads, seq_len = 256, 8, 4
    attn = QuantCB_Attention(d_model, n_heads)
    attn.eval() # Disable dropout if added later

    # Create two identical sequences
    x1 = torch.randn(1, seq_len, d_model)
    x2 = x1.clone()

    # Modify ONLY the last token in x2 (the "future")
    x2[0, -1, :] = torch.randn(d_model)

    with torch.no_grad():
        out1 = attn(x1)
        out2 = attn(x2)

    # Check first 3 tokens (the "past")
    # If causal masking works, the change at index 3 cannot affect indices 0-2
    past_diff = torch.abs(out1[0, :3, :] - out2[0, :3, :]).max().item()

    print(f"Max deviation in past tokens: {past_diff:.10f}")
    
    if past_diff == 0:
        print("✅ SUCCESS: The past is isolated from the future.")
    else:
        print("❌ FAILED: Information leaked backward through the graph.")

if __name__ == "__main__":
    test_causal_integrity()