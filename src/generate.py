import torch
import os
from models.quantcb_model import QuantCB_Model
from models.quantcb_engine import QuantCB_Engine
from tokenizer_basic import QuantCB_Tokenizer

def load_model_weights(model, checkpoint_path, device, is_fp8=False):
    """Handles loading either standard FP32 or Quantized FP8 weights."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Could not find checkpoint at {checkpoint_path}")

    # Use weights_only=True for security/stability unless you have custom classes saved
    if is_fp8:
        print(f"\n--- Loading Optimized FP8 Checkpoint: {checkpoint_path} ---")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        q_weights = checkpoint['weights']
        scales_dict = checkpoint['scales']
        
        dequantized_state_dict = {}
        for name, param in q_weights.items():
            scale_key = f"{name}_scales"
            if scale_key in scales_dict:
                param = param.to(device)
                scale_tensor = scales_dict[scale_key].to(device)
                original_shape = param.shape
                num_groups = scale_tensor.numel()
                group_size = param.numel() // num_groups
                
                param_float = param.view(-1, group_size).float()
                scale_expanded = scale_tensor.view(-1, 1)
                
                dequantized = (param_float * scale_expanded).view(original_shape)
                dequantized_state_dict[name] = dequantized
            else:
                dequantized_state_dict[name] = param.to(device)
        
        model.load_state_dict(dequantized_state_dict, strict=False)
    else:
        print(f"\n--- Loading Base FP32 Checkpoint: {checkpoint_path} ---")
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        # Standard checkpoints might be wrapped in a 'model_state_dict' key
        if 'model_state_dict' in state_dict:
            model.load_state_dict(state_dict['model_state_dict'], strict=True)
        else:
            model.load_state_dict(state_dict, strict=True)
    
    model.eval()
    return model

def generate():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir) 
    tok_path = os.path.join(project_root, "modelOutput", "tokenizer.json")

    print("Select Model Version:")
    print("[1] MTP (Standard FP32)")
    print("[2] Optimized (FP8)")
    choice = input("Enter 1 or 2: ").strip()

    if choice == '2':
        filename = "quantcb_fp8.pth"
        is_fp8 = True
    else:
        filename = "quantcb_final.pth"
        is_fp8 = False

    model_path = os.path.join(project_root, "modelOutput", filename)

    tokenizer = QuantCB_Tokenizer()
    if not os.path.exists(tok_path):
        print(f"Error: Tokenizer not found at {tok_path}")
        return
    tokenizer.load(tok_path)
    
    vocab_size = 2048 # Keeping this consistent with your training script
    
    # FIXED: Parameters updated to match the checkpoint's internal shapes
    raw_model = QuantCB_Model(
        vocab_size=vocab_size, 
        d_model=384,      # Matches checkpoint size mismatch error
        n_layers=6,       # Matches checkpoint layer count (0 through 5)
        d_ff=1024,        # Standard for your architecture
        n_heads=8,        # Standard for your d_model
        latent_dim=128, 
        head_dim=64,
        num_experts=8,    
        top_k=2           
    ).to(device)

    try:
        raw_model = load_model_weights(raw_model, model_path, device, is_fp8=is_fp8)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    engine = QuantCB_Engine(raw_model)

    context = torch.zeros((1, 1), dtype=torch.long, device=device) 
    
    print(f"\nGenerating 300 tokens (MLA + MoE + MTP Architecture)...\n" + "="*40)
    
    with torch.no_grad():
        # Using temperature and top_p here is recommended for better quality
        generated_ids = engine.generate(context, max_new_tokens=300, temperature=0.8)[0].tolist()
    
    output_text = tokenizer.decode(generated_ids)
    print(output_text)
    print("\n" + "="*40)

if __name__ == "__main__":
    generate()