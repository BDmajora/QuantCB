import os
import torch

def quantize_to_int8(tensor):
    """Symmetric quantization to int8."""
    if not isinstance(tensor, torch.Tensor):
        return tensor
    
    # Calculate scale: mapping the max absolute value to 127
    max_val = tensor.abs().max().item()
    if max_val == 0:
        return tensor, 1.0
        
    scale = max_val / 127.0
    
    # Quantize: divide by scale, round, and clamp to int8 limits
    q_tensor = (tensor / scale).round().clamp(-128, 127).to(torch.int8)
    
    return q_tensor, scale

def run_quantization():
    # Dynamic pathing to /modelOutput relative to project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, "modelOutput")
    
    checkpoint_path = os.path.join(output_dir, 'quantcb_base.pth')
    output_path = os.path.join(output_dir, 'quantcb_int8.pth')
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Could not find base checkpoint at {checkpoint_path}")
        return

    print(f"--- Loading FP32 Checkpoint: {checkpoint_path} ---")
    state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    quantized_state_dict = {}
    scales = {}

    for name, param in state_dict.items():
        # Only quantize weights; skip biases and LayerNorm for stability
        if 'weight' in name and 'ln' not in name:
            q_weight, scale = quantize_to_int8(param)
            quantized_state_dict[name] = q_weight
            scales[f"{name}_scale"] = scale
            print(f"Quantized {name:40} | Scale: {scale:.6f}")
        else:
            quantized_state_dict[name] = param

    # Save the compressed weights and the scales needed for dequantization
    torch.save({'weights': quantized_state_dict, 'scales': scales}, output_path)
    
    orig_size = os.path.getsize(checkpoint_path) / (1024*1024)
    quant_size = os.path.getsize(output_path) / (1024*1024)
    
    print(f"\nOptimization Complete.")
    print(f"Original Size (FP32): {orig_size:.2f} MB")
    print(f"Compressed Size (INT8): {quant_size:.2f} MB")
    print(f"Compression Ratio: {orig_size/quant_size:.2f}x")

if __name__ == "__main__":
    run_quantization()