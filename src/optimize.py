import os
import torch

def quantize_to_fp8_fine_grained(tensor, group_size=128):
    """
    Fine-grained FP8 (E4M3) quantization using group-wise scaling.
    Designed for better precision and DeepGEMM compatibility on 8-bit hardware.
    """
    if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
        return tensor, None
    
    # FP8 E4M3 maximum representable value
    FP8_MAX = 448.0
    
    original_shape = tensor.shape
    
    # Fallback to per-tensor if the tensor size isn't cleanly divisible by the group size
    if tensor.numel() % group_size != 0:
        group_size = tensor.numel()
        
    # Reshape into groups for fine-grained scaling
    grouped_tensor = tensor.view(-1, group_size)
    
    # Calculate absolute maximum per group (clamped to avoid division by zero)
    max_vals = grouped_tensor.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-12)
    
    # Compute scales per group
    scales = max_vals / FP8_MAX
    
    # Quantize: scale down and cast to native PyTorch FP8 E4M3
    # PyTorch 2.1+ natively supports float8_e4m3fn conversions
    q_tensor = (grouped_tensor / scales).to(torch.float8_e4m3fn)
    
    # Reshape weight back to original, and keep scales as a flattened vector
    return q_tensor.view(original_shape), scales.view(-1)

def run_quantization():
    # Dynamic pathing relative to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    output_dir = os.path.join(project_root, "modelOutput")
    
    checkpoint_path = os.path.join(output_dir, 'quantcb_final.pth')
    # Updated output path for FP8
    output_path = os.path.join(output_dir, 'quantcb_fp8.pth')
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Could not find base checkpoint at {checkpoint_path}")
        return

    print(f"--- Loading FP32 Checkpoint: {checkpoint_path} ---")
    # Load the state dict trained with MLA + MoE
    state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    
    quantized_weights = {}
    scales_dict = {}

    print(f"Quantizing layers using Fine-Grained FP8 Scaling...")
    for name, param in state_dict.items():
        # Quantize heavy linear weights (Attention, MoE Experts, Router, and LM Head)
        # We skip LayerNorm (ln) and biases to maintain numerical stability
        if 'weight' in name and 'ln' not in name and 'embedding' not in name:
            q_weight, block_scales = quantize_to_fp8_fine_grained(param, group_size=128)
            quantized_weights[name] = q_weight
            
            # Store the fine-grained scales for this tensor
            if block_scales is not None:
                scales_dict[f"{name}_scales"] = block_scales
            
            # Print feedback for expert layers specifically
            if 'moe.experts' in name:
                # Truncate long names for cleaner console output
                short_name = name.replace('blocks.', 'b').replace('.moe.experts.', '.e')
                print(f"Compressed {short_name:30} | Scale Groups: {block_scales.numel()}")
        else:
            # Keep biases, embeddings, and norms in FP32
            quantized_weights[name] = param

    # Save the compressed weight dictionary and the scales for DeepGEMM-style execution
    torch.save({
        'weights': quantized_weights, 
        'scales': scales_dict,
        'metadata': {
            'architecture': 'MLA+MoE',
            'precision': 'FP8-FineGrained',
            'target_engine': 'DeepGEMM',
            'group_size': 128
        }
    }, output_path)
    
    # Calculate file sizes
    orig_size = os.path.getsize(checkpoint_path) / (1024*1024)
    quant_size = os.path.getsize(output_path) / (1024*1024)
    
    print(f"\nOptimization Complete.")
    print(f"Original FP32 Size: {orig_size:.2f} MB")
    print(f"Compressed FP8 Size: {quant_size:.2f} MB")
    print(f"Total Model Reduction: {orig_size/quant_size:.2f}x")
    print(f"Note: Weights saved as torch.float8_e4m3fn with group-wise scaling.")

if __name__ == "__main__":
    run_quantization()