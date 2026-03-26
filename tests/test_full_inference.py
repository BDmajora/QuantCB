import torch
from models.quantcb_model import QuantCB_Model

def test_full_inference():
    print("--- Running Full QuantCB Model Inference (MLA) Test ---")
    
    # Configuration to match your optimized 16M parameter goal
    vocab_size = 50257 
    d_model = 256
    n_heads = 8
    n_layers = 4
    latent_dim = 128
    head_dim = 64
    
    # Initialize with the new architectural parameters
    model = QuantCB_Model(
        vocab_size=vocab_size, 
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        latent_dim=latent_dim,
        head_dim=head_dim
    )
    model.eval()

    # Simulate a batch of 2 sequences, each 16 tokens long
    dummy_input = torch.randint(0, vocab_size, (2, 16))
    
    with torch.no_grad():
        # The model now internally handles the causal mask
        logits, _ = model(dummy_input)

    print(f"Input IDs Shape:  {dummy_input.shape}")
    print(f"Output Logits Shape: {logits.shape}")

    # Verification
    assert logits.shape == (2, 16, vocab_size), f"Expected (2, 16, {vocab_size}), got {logits.shape}"
    
    # Check for NaN (common when attention scaling is off)
    assert not torch.isnan(logits).any(), "Output contains NaNs!"
    
    print("SUCCESS: Full MLA-based model pipeline operational.")

if __name__ == "__main__":
    try:
        test_full_inference()
    except Exception as e:
        print(f"FAILED: {e}")