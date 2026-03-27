import os
import torch

def quantize_to_int8(tensor):
    """Symmetric quantization to int8: maps the max absolute value to 127."""
    if not isinstance(tensor, torch.Tensor):
        return tensor
    
    # Calculate scale for symmetric quantization
    max_val = tensor.abs().max().item()
    if max_val == 0:
        return tensor, 1.0
        
    scale = max_val / 127.0
    
    # Quantize: scale, round, and clamp to signed 8-bit range
    q_tensor = (tensor / scale).round().clamp(-128, 127).to(torch.int8)
    
    return q_tensor, scale

def run_quantization():
    # Dynamic pathing relative to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    output_dir = os.path.join(project_root, "modelOutput")
    
    checkpoint_path = os.path.join(output_dir, 'quantcb_base.pth')
    output_path = os.path.join(output_dir, 'quantcb_int8.pth')
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Could not find base checkpoint at {checkpoint_path}")
        return

    print(f"--- Loading FP32 Checkpoint: {checkpoint_path} ---")
    # Load the state dict trained with MLA + MoE
    state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    
    quantized_weights = {}
    scales = {}

    print(f"Quantizing layers...")
    for name, param in state_dict.items():
        # Quantize heavy linear weights (Attention, MoE Experts, Router, and LM Head)
        # We skip LayerNorm (ln) and biases to maintain numerical stability in the small model
        if 'weight' in name and 'ln' not in name and 'embedding' not in name:
            q_weight, scale = quantize_to_int8(param)
            quantized_weights[name] = q_weight
            scales[f"{name}_scale"] = scale
            
            # Print feedback for expert layers specifically
            if 'moe.experts' in name:
                # Truncate long names for cleaner console output
                short_name = name.replace('blocks.', 'b').replace('.moe.experts.', '.e')
                print(f"Compressed {short_name:30} | Scale: {scale:.6f}")
        else:
            # Keep biases, embeddings, and norms in FP32
            quantized_weights[name] = param

    # Save the compressed weight dictionary and the scales for dequantization-on-load
    torch.save({
        'weights': quantized_weights, 
        'scales': scales,
        'metadata': {
            'architecture': 'MLA+MoE',
            'precision': 'INT8-Symmetric'
        }
    }, output_path)
    
    orig_size = os.path.getsize(checkpoint_path) / (1024*1024)
    quant_size = os.path.getsize(output_path) / (1024*1024)
    
    print(f"\nOptimization Complete.")
    print(f"Original FP32 Size: {orig_size:.2f} MB")
    print(f"Compressed INT8 Size: {quant_size:.2f} MB")
    print(f"Total Model Reduction: {orig_size/quant_size:.2f}x")

if __name__ == "__main__":
    run_quantization()