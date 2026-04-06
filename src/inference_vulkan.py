import os
import numpy as np
import iree.runtime as ireert
from pathlib import Path

# Direct import from same directory
from config import *

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

def test_inference_vulkan():
    print("--- 2026 IREE-Runtime: Vulkan/SPIR-V Hardware Test ---")
    
    # 1. Locate the baked binary
    vmfb_path = PROJECT_ROOT / OUTPUT_DIR / "quantcb_vulkan.vmfb"
    
    if not vmfb_path.exists():
        print(f"ERROR: {vmfb_path} not found.")
        return

    # 2. Setup the runtime config
    config = ireert.Config("vulkan")
    
    print("Loading VMFB into GPU Memory (using mmap for alignment)...")
    # Using mmap ensures zero-copy loading and satisfies IREE alignment requirements
    vm_module = ireert.VmModule.mmap(config.vm_instance, str(vmfb_path))
    
    # 3. Create the system context and bind the module
    ctx = ireert.SystemContext(config=config)
    ctx.add_vm_module(vm_module)
    
    # Access the module (default export name is 'module')
    bound_module = ctx.modules.module

    # 4. Prepare Input (Batch x Block)
    # The model expects i64 (int64) for token indices
    input_item = np.random.randint(0, VOCAB_SIZE, size=(BATCH_SIZE, BLOCK_SIZE), dtype=np.int64)
    print(f"Input Generated: Shape {input_item.shape} (dtype: int64)")
    
    # 5. Execute SPIR-V Kernel Dispatch
    print("Dispatching SPIR-V Kernels to RX 6800...")
    
    try:
        # Call the entry point (main)
        results = bound_module.main(input_item)
        
        # Results structure: [logits, presents, l_aux]
        logits = results[0]
        l_aux = results[2]

        print("\n" + "="*30)
        print("INFERENCE SUCCESSFUL")
        print("="*30)
        
        # Handle shape display as a standard list/tuple
        actual_shape = tuple(logits.shape)
        print(f"Output Logits Shape: {actual_shape}")
        
        # Safe Hardware Logging
        try:
            print(f"Target Hardware: {config.device.name}")
        except AttributeError:
            print("Target Hardware: Vulkan/SPIR-V Device")

        # 6. Validation against config constants
        expected_shape = (BATCH_SIZE, BLOCK_SIZE, VOCAB_SIZE)
        if actual_shape == expected_shape:
            print("SUCCESS: Output shapes and Kernel Dispatch verified.")
            print(f"MoE Auxiliary Loss (Load Balance): {l_aux}")
        else:
            print(f"WARNING: Shape mismatch! Expected {expected_shape}, got {actual_shape}")

    except Exception as e:
        print(f"\nRuntime Error during execution: {e}")

if __name__ == "__main__":
    test_inference_vulkan()