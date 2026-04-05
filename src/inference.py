import os
import torch
from quant_core import get_rotation_matrix, polar_to_cartesian
# Assuming QuantCB_Model is in a models/ folder relative to this file
from models.quantcb_model import QuantCB_Model 

def load_quantized_model(model, path):
    print(f"--- Loading Polar-QJL Weights from {path} ---")
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    weights = checkpoint['weights']
    sd = model.state_dict()
    
    for name, param in sd.items():
        if name not in weights: continue
        data = weights[name]
        
        if data['type'] == 'polar_qjl':
            # 1. Dequantize Polar Base
            angle_scale = 3.14159 / (2**4 - 1)
            angles = data['angles_q'].to(torch.float32) * angle_scale
            W_rot = polar_to_cartesian(data['r'].to(torch.float32), angles)
            
            # 2. Add 1-bit QJL Residual
            W_rot += data['res_sign'].to(torch.float32) * data['res_scale']
            
            # 3. Inverse Rotation
            R = get_rotation_matrix(data['shape'][1], data['seed'])
            sd[name] = W_rot @ R.T
        else:
            sd[name] = data['weight']
            
    model.load_state_dict(sd)
    return model

def generate_test():
    # ... (Your model config from before) ...
    model = QuantCB_Model(vocab_size=50257, d_model=256, n_heads=8, d_ff=1024, n_layers=4)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    quant_path = os.path.join(project_root, "modelOutput", "quantcb_turbo_polar.pth")

    if os.path.exists(quant_path):
        model = load_quantized_model(model, quant_path)
        model.eval()
        prompt = torch.randint(0, 50257, (1, 5))
        with torch.no_grad():
            output = model(prompt)
            logits = output[0] if isinstance(output, tuple) else output
            print(f"Inference verified. Logit Max: {logits.max().item():.4f}")

if __name__ == "__main__":
    generate_test()