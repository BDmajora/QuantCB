import torch
import torch.nn as nn
from models.quantcb_model import QuantCB_Model

def deep_scan_offenders(model):
    print("\n" + "!"*60)
    print("DEEP PARAMETER SCAN: FINDING LIFTED TENSORS")
    print("!"*60)
    
    found = 0
    # Iterate through every sub-module in the architecture
    for mod_name, mod in model.named_modules():
        # Check every attribute assigned to that module
        for attr_name, attr_val in mod.__dict__.items():
            # We are looking for Tensors that require gradients 
            # but were never wrapped in nn.Parameter()
            if isinstance(attr_val, torch.Tensor) and attr_val.requires_grad:
                if not isinstance(attr_val, torch.nn.Parameter):
                    print(f"OFFENDER FOUND:")
                    print(f"   Module:    {mod_name if mod_name else 'Root'}")
                    print(f"   Attribute: {attr_name}")
                    print(f"   Shape:     {list(attr_val.shape)}")
                    print(f"   Fix: Wrap this in nn.Parameter() in {mod.__class__.__name__}.__init__")
                    print("-" * 30)
                    found += 1
                    
    if found == 0:
        print("SUCCESS: No raw tensors found. The issue may be complex aliasing.")
    else:
        print(f"TOTAL OFFENDERS DETECTED: {found}")
    print("!"*60 + "\n")

if __name__ == "__main__":
    # Initialize with your specific hyperparameters
    model = QuantCB_Model(
        vocab_size=50257, 
        d_model=256, 
        n_layers=6,
        num_experts=8, 
        top_k=2
    )
    deep_scan_offenders(model)