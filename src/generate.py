import os
import torch
from config import *
from models.quantcb_model import QuantCB_Model
from models.ouro_engine import Ouro_Engine 
from tokenizer import Tokenizer

def load_model_weights(model, checkpoint_path, device, is_fp8=False):
    """Handles loading either standard FP32 or Quantized FP8 weights."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Could not find checkpoint at {checkpoint_path}")

    # Use weights_only=True for security/stability
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
        
        # Check for wrapped 'model_state_dict' from trainer saving logic
        if 'model_state_dict' in state_dict:
            model.load_state_dict(state_dict['model_state_dict'], strict=True)
        else:
            model.load_state_dict(state_dict, strict=True)
    
    model.eval()
    return model

def generate():
    device = DEVICE # Using the device from your config
    
    # Pathing aligned with your recent trainer updates
    tok_path = os.path.join(OUTPUT_DIR, "quantcb_tokenizer.json")

    print("\nSelect Model Version:")
    print("[1] MTP (Standard FP32)")
    print("[2] Optimized (FP8)")
    choice = input("Enter 1 or 2: ").strip()

    if choice == '2':
        filename = "quantcb_fp8.pth"
        is_fp8 = True
    else:
        # Check for the checkpoint name used in your new modular trainer
        filename = "quantcb_ckpt.pth" if os.path.exists(os.path.join(OUTPUT_DIR, "quantcb_ckpt.pth")) else "quantcb_final.pth"
        is_fp8 = False

    model_path = os.path.join(OUTPUT_DIR, filename)

    # Use the new fixed Tokenizer
    tokenizer = Tokenizer()
    if not tokenizer.load(tok_path):
        print(f"Error: Tokenizer not found at {tok_path}. Please run training first.")
        return
    
    # Initializing the model with your specific architectural requirements
    raw_model = QuantCB_Model(
        vocab_size=VOCAB_SIZE, 
        d_model=384,      
        n_layers=6,       
        d_ff=1024,        
        n_heads=8,        
        num_experts=8,    
        top_k=2           
    ).to(device)

    try:
        raw_model = load_model_weights(raw_model, model_path, device, is_fp8=is_fp8)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Wrap in the Ouro Engine for the recursive thinking logic
    engine = Ouro_Engine(
        raw_model, 
        max_loops=MAX_LOOPS, 
        exit_threshold=EXIT_THRESHOLD
    ).to(device)

    # Prompting
    prompt = input("\nEnter prompt (or press Enter for default): ").strip()
    if not prompt:
        seed_str = f"{TAGS['truth']}{TAGS['stories']} Once upon a time"
    else:
        seed_str = prompt

    context_ids = tokenizer.encode(seed_str)
    context = torch.tensor([context_ids], dtype=torch.long, device=device)
    
    print(f"\nGenerating... (Ouro Logic: {MAX_LOOPS} Loops Max | {EXIT_THRESHOLD} Entropy Exit)\n" + "="*40)
    
    with torch.no_grad():
        # Temperature 0.8 provides a good balance of creativity and coherence
        generated_ids = engine.generate(context, max_new_tokens=300, temperature=0.8)
    
    output_text = tokenizer.decode(generated_ids[0].tolist())
    print(output_text)
    print("\n" + "="*40)

if __name__ == "__main__":
    generate()