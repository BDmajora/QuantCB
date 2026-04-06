import os
import torch
import numpy as np
import iree.runtime as ireert
import iree.turbine.aot as aot  # The correct import for your reqs

# Your requested imports
from config import *
from models.quantcb_model import QuantCB_Model
from models.ouro_engine import Ouro_Engine 
from tokenizer import Tokenizer

def load_model_weights(model, checkpoint_path, device, is_fp8=False):
    """Handles loading either standard FP32 or Quantized FP8 weights."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Could not find checkpoint at {checkpoint_path}")

    # For AOT Compilation, we always load to CPU first
    load_device = "cpu" 
    
    if is_fp8:
        print(f"\n--- Loading Optimized FP8 Checkpoint: {checkpoint_path} ---")
        checkpoint = torch.load(checkpoint_path, map_location=load_device, weights_only=True)
        q_weights = checkpoint['weights']
        scales_dict = checkpoint['scales']
        
        dequantized_state_dict = {}
        for name, param in q_weights.items():
            scale_key = f"{name}_scales"
            if scale_key in scales_dict:
                scale_tensor = scales_dict[scale_key]
                original_shape = param.shape
                num_groups = scale_tensor.numel()
                group_size = param.numel() // num_groups
                
                param_float = param.view(-1, group_size).float()
                scale_expanded = scale_tensor.view(-1, 1)
                
                dequantized = (param_float * scale_expanded).view(original_shape)
                dequantized_state_dict[name] = dequantized
            else:
                dequantized_state_dict[name] = param
        
        model.load_state_dict(dequantized_state_dict, strict=False)
    else:
        print(f"\n--- Loading Base FP32 Checkpoint: {checkpoint_path} ---")
        state_dict = torch.load(checkpoint_path, map_location=load_device, weights_only=True)
        if 'model_state_dict' in state_dict:
            model.load_state_dict(state_dict['model_state_dict'], strict=True)
        else:
            model.load_state_dict(state_dict, strict=True)
    
    model.eval()
    return model

def compile_vulkan_module(engine, example_input, vmfb_path):
    """Compiles the Ouro Engine logic into a Vulkan binary."""
    print(f"--- Compiling for Vulkan (RX 6800)... ---")
    
    # Trace and compile
    # This turns your PyTorch math into SPIR-V code for the AMD GPU
    compiled_module = aot.export(
        engine,
        args=(example_input,),
        target="vulkan",
        module_name="ouro_vulkan"
    )
    
    compiled_module.save(vmfb_path)
    print(f"Compiled: {vmfb_path}")

def generate():
    tok_path = os.path.join(OUTPUT_DIR, "quantcb_tokenizer.json")
    vmfb_path = os.path.join(OUTPUT_DIR, "quantcb_vulkan.vmfb")

    print("\nSelect Model Version:")
    print("[1] MTP (Standard FP32)")
    print("[2] Optimized (FP8)")
    choice = input("Enter 1 or 2: ").strip()

    is_fp8 = (choice == '2')
    filename = "quantcb_fp8.pth" if is_fp8 else ("quantcb_ckpt.pth" if os.path.exists(os.path.join(OUTPUT_DIR, "quantcb_ckpt.pth")) else "quantcb_final.pth")
    model_path = os.path.join(OUTPUT_DIR, filename)

    tokenizer = Tokenizer()
    if not tokenizer.load(tok_path):
        print(f"Error: Tokenizer not found.")
        return
    
    # 1. Initialize Model and Engine (Keep on CPU for Tracing)
    raw_model = QuantCB_Model(
        vocab_size=VOCAB_SIZE, 
        d_model=384, n_layers=6, d_ff=1024, 
        n_heads=8, num_experts=8, top_k=2           
    )

    try:
        raw_model = load_model_weights(raw_model, model_path, "cpu", is_fp8=is_fp8)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    engine = Ouro_Engine(raw_model, max_loops=MAX_LOOPS, exit_threshold=EXIT_THRESHOLD)

    # 2. Check for Compiled Vulkan Binary
    if not os.path.exists(vmfb_path):
        # We need a dummy input of the correct shape to trace the graph
        dummy_input = torch.zeros((1, BLOCK_SIZE), dtype=torch.long)
        compile_vulkan_module(engine, dummy_input, vmfb_path)

    # 3. Setup IREE Runtime for RX 6800
    config = ireert.Config("vulkan")
    vmfb_module = ireert.VmModule.mmap(config.vm_instance, vmfb_path)
    hal_module = ireert.create_hal_module(config.vm_instance, config.device)
    
    # Load context
    ctx = ireert.SystemContext(config=config)
    ctx.add_vm_module(vmfb_module)

    # 4. Prompting
    prompt = input("\nEnter prompt: ").strip()
    seed_str = prompt if prompt else "Once upon a time"
    context_ids = tokenizer.encode(seed_str)
    
    # Pad or truncate to match the compiled BLOCK_SIZE
    if len(context_ids) < BLOCK_SIZE:
        context_ids = context_ids + [0] * (BLOCK_SIZE - len(context_ids))
    context_ids = context_ids[:BLOCK_SIZE]
    
    # Move tensor to IREE Device Array
    input_array = ireert.asdevicearray(config.device, np.array([context_ids], dtype=np.int64))

    print(f"\nGenerating on Vulkan... \n" + "="*40)
    
    # Execute the compiled 'forward' method
    # Note: 'main' is the default exported name in Turbine AOT
    result_array = ctx.modules.ouro_vulkan.main(input_array)
    
    # Convert back to host
    output_ids = torch.tensor(result_array.to_host())
    output_text = tokenizer.decode(output_ids[0].tolist())
    
    print(output_text)
    print("\n" + "="*40)

if __name__ == "__main__":
    generate()