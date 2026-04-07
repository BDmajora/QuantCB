import torch
from models.quantcb_model import QuantCB_Model

def check_weight_aliasing(model):
    print("\n" + "="*60)
    print("WEIGHT ALIASING AND IDENTITY SCAN")
    print("="*60)
    
    weights_seen = {}
    duplicates = 0
    
    for name, param in model.named_parameters():
        # Use the memory address of the underlying data as the unique ID
        data_ptr = param.data_ptr()
        
        if data_ptr in weights_seen:
            first_name = weights_seen[data_ptr]
            print(f"ALIAS DETECTED:")
            print(f"   Address: {data_ptr}")
            print(f"   Name 1:  {first_name}")
            print(f"   Name 2:  {name}")
            print(f"   Action:  These must be the EXACT same object instance.")
            print("-" * 30)
            duplicates += 1
        else:
            weights_seen[data_ptr] = name
            
    if duplicates == 0:
        print("SUCCESS: No memory aliasing detected.")
    else:
        print(f"TOTAL ALIASES FOUND: {duplicates}")
    print("="*60 + "\n")

if __name__ == "__main__":
    model = QuantCB_Model(vocab_size=50257)
    check_weight_aliasing(model)