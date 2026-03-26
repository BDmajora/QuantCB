import torch
from models.quantcb_model import QuantCB_Model

def test_full_inference():
    print("--- Running Full QuantCB Model Inference Test ---")
    vocab_size = 50257 # Match standard GPT-2 vocab size
    model = QuantCB_Model(vocab_size=vocab_size, n_layers=4)
    model.eval()

    # Simulate a batch of 2 sequences, each 16 tokens long
    dummy_input = torch.randint(0, vocab_size, (2, 16))
    
    with torch.no_grad():
        logits, _ = model(dummy_input)

    print(f"Input IDs Shape: {dummy_input.shape}")
    print(f"Output Logits Shape: {logits.shape}")

    assert logits.shape == (2, 16, vocab_size)
    print("SUCCESS: Full model pipeline operational.")

if __name__ == "__main__":
    test_full_inference()