import os
import torch
from quant_core import get_rotation_matrix, cartesian_to_polar

def turbo_polar_compress(tensor, name, bits=4):
    """Hybrid PolarQuant + QJL Residual compression."""
    # Ensure we only process 2D floating point weight tensors
    if not isinstance(tensor, torch.Tensor) or tensor.dim() != 2 or not tensor.is_floating_point():
        return {'type': 'unquantized', 'weight': tensor.cpu() if isinstance(tensor, torch.Tensor) else tensor}
    
    out_f, in_f = tensor.shape
    seed = sum(ord(c) for c in name) % 1000000
    
    # 1. Preconditioning (Random Rotation)
    R = get_rotation_matrix(in_f, seed, device=tensor.device)
    W_rot = tensor @ R
    
    # 2. Polar Transformation
    r, angles = cartesian_to_polar(W_rot)
    
    # Quantize Angles (0 to pi)
    angle_scale = 3.14159 / (2**bits - 1)
    angles_q = torch.round(angles / angle_scale).to(torch.int8)
    
    # 3. 1-bit QJL Residual (Unbiased estimator component)
    # Reconstruct base to find residual
    angles_deq = angles_q.to(torch.float32) * angle_scale
    from quant_core import polar_to_cartesian # Local import to avoid circularity if needed
    W_base = polar_to_cartesian(r, angles_deq)
    
    residual = W_rot - W_base
    res_sign = torch.where(residual >= 0, 1, -1).to(torch.int8)
    res_scale = residual.abs().mean().item() # Global scalar = Zero Overhead
    
    return {
        'type': 'polar_qjl',
        'r': r.to(torch.float16).cpu(),
        'angles_q': angles_q.cpu(),
        'res_sign': res_sign.cpu(),
        'res_scale': res_scale,
        'seed': seed,
        'shape': (out_f, in_f)
    }

def run_quantization():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    output_dir = os.path.join(project_root, "modelOutput")
    
    checkpoint_path = os.path.join(output_dir, 'quantcb_final.pth')
    output_path = os.path.join(output_dir, 'quantcb_turbo_polar.pth')
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: {checkpoint_path} not found.")
        return

    print(f"--- Loading FP32 Checkpoint ---")
    state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    new_dict = {}

    print(f"Applying PolarQuant + QJL Residuals...")
    for name, param in state_dict.items():
        # Check if it's a tensor and specifically a 2D weight matrix
        is_weight = isinstance(param, torch.Tensor) and param.dim() == 2 and 'weight' in name and 'ln' not in name
        
        if is_weight:
            new_dict[name] = turbo_polar_compress(param, name)
            if 'moe' in name:
                print(f"Compressed Expert: {name}")
        else:
            # Handle non-tensor metadata or 1D/3D params safely
            safe_val = param.cpu() if isinstance(param, torch.Tensor) else param
            new_dict[name] = {'type': 'unquantized', 'weight': safe_val}

    torch.save({'weights': new_dict, 'metadata': {'method': 'PolarQuant+QJL'}}, output_path)
    
    orig_size = os.path.getsize(checkpoint_path) / (1024**2)
    quant_size = os.path.getsize(output_path) / (1024**2)
    print(f"\nSuccess! Size: {orig_size:.2f}MB -> {quant_size:.2f}MB ({orig_size/quant_size:.2f}x)")

if __name__ == "__main__":
    run_quantization()