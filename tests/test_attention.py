import torch
# No more relative path mess—import directly from the root-level models folder
from models.mla_original import MLA_Attention 

def test_mla_logic():
    print("--- Running MLA Attention Test from /tests/ ---")
    
    # 1. Configuration (16M parameter specs)
    batch_size = 2
    seq_len = 8
    d_model = 256
    n_heads = 8
    latent_dim = 128
    head_dim = 64
    
    # 2. Init Layer
    # Using the updated MLA_Attention class
    attn = MLA_Attention(
        d_model=d_model, 
        n_heads=n_heads, 
        latent_dim=latent_dim, 
        head_dim=head_dim
    )
    
    # 3. Create Dummy Tensor and Causal Mask
    x = torch.randn(batch_size, seq_len, d_model)
    # Masking is required for generative Transformer logic
    mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
    
    try:
        # 4. Forward Pass
        output = attn(x, mask=mask)
        
        print(f"Input Shape:  {x.shape}")
        print(f"Output Shape: {output.shape}")
        
        # 5. Verify Structure
        assert output.shape == x.shape, f"Shape Mismatch: Expected {x.shape}, got {output.shape}"
        assert not torch.isnan(output).any(), "NaN detected in attention output!"
        
        print("SUCCESS: MLA Attention logic is operational from /tests/ folder.")
        
    except Exception as e:
        print(f"FAILED: {e}")

if __name__ == "__main__":
    test_mla_logic()