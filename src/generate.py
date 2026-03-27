import torch
import os
from models.quantcb_model import QuantCB_Model
from tokenizer_basic import QuantCB_Tokenizer

def load_model_weights(model, checkpoint_path, device, is_int8=False):
    """Handles loading either standard FP32 or Quantized INT8 weights."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Could not find checkpoint at {checkpoint_path}")

    if is_int8:
        print(f"\n--- Loading Optimized INT8 Checkpoint: {checkpoint_path} ---")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        q_weights = checkpoint['weights']
        scales = checkpoint['scales']
        
        # Dequantize weights back to FP32 for inference
        dequantized_state_dict = {}
        for name, param in q_weights.items():
            if f"{name}_scale" in scales:
                dequantized_state_dict[name] = param.float() * scales[f"{name}_scale"]
            else:
                dequantized_state_dict[name] = param 
        
        model.load_state_dict(dequantized_state_dict)
    else:
        print(f"\n--- Loading Base FP32 Checkpoint: {checkpoint_path} ---")
        # Use weights_only=True for security and performance
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
    
    model.eval()
    return model

def generate():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Pathing Setup
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir) 
    # Tokenizer is now stored in modelOutput
    tok_path = os.path.join(project_root, "modelOutput", "tokenizer.json")

    # Interactive Model Selection
    print("Select Model Version:")
    print("[1] Base (FP32)")
    print("[2] Optimized (INT8)")
    choice = input("Enter 1 or 2: ").strip()

    if choice == '2':
        filename = "quantcb_int8.pth"
        is_int8 = True
    else:
        filename = "quantcb_base.pth"
        is_int8 = False

    model_path = os.path.join(project_root, "modelOutput", filename)

    # 1. Setup Tokenizer
    tokenizer = QuantCB_Tokenizer()
    if not os.path.exists(tok_path):
        print(f"Error: Tokenizer not found at {tok_path}")
        return
    tokenizer.load(tok_path)
    
    # 2. Architecture (SYNCED WITH TRAINING: d_ff=512, experts=8, top_k=2)
    vocab_size = 2048 
    model = QuantCB_Model(
        vocab_size=vocab_size, 
        d_model=256, 
        n_heads=8, 
        d_ff=512,         # MATCHES YOUR LATEST TRAINING RUN
        n_layers=4,
        latent_dim=128, 
        head_dim=64,
        num_experts=8,    # NEW: MoE parameter
        top_k=2           # NEW: MoE parameter
    ).to(device)

    # 3. Load Weights
    try:
        model = load_model_weights(model, model_path, device, is_int8=is_int8)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 4. The Actual Generation
    # Start with a sequence of 0 (usually the [PAD] or [BOS] token)
    context = torch.zeros((1, 1), dtype=torch.long, device=device) 
    
    print(f"\nGenerating 300 tokens (MLA + MoE Engine)...\n" + "="*40)
    
    with torch.no_grad():
        # MLA Cache is handled internally within model.generate
        generated_ids = model.generate(context, max_new_tokens=300)[0].tolist()
    
    # 5. Decode
    output_text = tokenizer.decode(generated_ids)
    print(output_text)
    print("\n" + "="*40)

if __name__ == "__main__":
    generate()