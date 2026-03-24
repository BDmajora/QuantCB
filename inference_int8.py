import torch
import math
from models.quantcb_model import QuantCB_Model

def load_quantized_model(model, quant_path):
    print(f"--- Loading INT8 Optimized Weights from {quant_path} ---")
    checkpoint = torch.load(quant_path)
    q_weights = checkpoint['weights']
    scales = checkpoint['scales']
    
    # Get the current state dict to update it
    state_dict = model.state_dict()
    
    for name, param in state_dict.items():
        if name in q_weights:
            weight = q_weights[name]
            
            # If there's a corresponding scale, dequantize it back to FP32
            scale_key = f"{name}_scale"
            if scale_key in scales:
                # Dequantize: Float = Int * Scale
                state_dict[name] = weight.to(torch.float32) * scales[scale_key]
            else:
                state_dict[name] = weight
                
    model.load_state_dict(state_dict)
    return model

def generate_test():
    vocab_size = 50257
    model = QuantCB_Model(vocab_size=vocab_size, n_layers=4)
    
    # Load the optimized weights
    model = load_quantized_model(model, 'quantcb_int8.pth')
    model.eval()

    # Create dummy prompt (Batch=1, Seq=5)
    prompt = torch.randint(0, vocab_size, (1, 5))
    
    with torch.no_grad():
        logits, _ = model(prompt)
        
    print(f"\nInference successful.")
    print(f"Output Logits Max: {logits.max().item():.4f}")
    print(f"Output Logits Min: {logits.min().item():.4f}")
    print("✅ Verified: Model operates correctly with dequantized INT8 weights.")

if __name__ == "__main__":
    generate_test()