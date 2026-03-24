import torch
# Since attention.py is in /models/, we use the dot notation
from models.attention import QuantCB_Attention

def test_mha_logic():
    print("--- Running Attention Smoke Test (Subdirectory) ---")
    
    # 1. Configuration
    batch_size = 2
    seq_len = 8
    d_model = 256
    n_heads = 8
    
    # 2. Init Layer
    attn = QuantCB_Attention(d_model=d_model, n_heads=n_heads)
    
    # 3. Create Dummy Tensor
    x = torch.randn(batch_size, seq_len, d_model)
    
    try:
        # 4. Forward Pass
        output = attn(x)
        print(f"Input Shape:  {x.shape}")
        print(f"Output Shape: {output.shape}")
        
        # 5. Verify Math
        assert output.shape == x.shape, "Shape Mismatch Error!"
        print("✅ SUCCESS: models/attention.py is working correctly.")
        
    except Exception as e:
        print(f"❌ FAILED: {e}")

if __name__ == "__main__":
    test_mha_logic()