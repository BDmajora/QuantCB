import torch
import os
# We use the Turbine AOT (Ahead-of-Time) compiler 
# This is the 2026 standard for IREE + PyTorch
from iree.turbine import aot 
from config import *
from models.quantcb_model import QuantCB_Model

def compile_for_vulkan():
    print("--- 2026 IREE-Turbine: General Vulkan Bake Starting ---")
    
    # 1. Setup Model (Still on CPU for the bake)
    model = QuantCB_Model(
        vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_layers=N_LAYERS, 
        num_experts=NUM_EXPERTS, top_k=TOP_K
    )
    
    # 2. Define the input signature
    # Using static shapes here allows for the most efficient SPIR-V generation.
    class TrainModule(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            
        def forward(self, x):
            # Compiles all 3 returns (logits, presents, l_aux) into the binary
            return self.model(x)

    # 3. Export to IREE/Vulkan via Turbine
    print("Exporting PyTorch Graph to MLIR...")
    example_input = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LENGTH))
    export_output = aot.export(TrainModule(model), example_input)
    
    # 4. Compile to VMFB for General Vulkan Support
    print("Compiling General SPIR-V Kernels (Hardware-Agnostic)...")
    output_path = os.path.join(OUTPUT_DIR, "quantcb_vulkan.vmfb")
    
    # ---> THE FIX IS HERE <---
    # 'vulkan-spirv' is the correct compiler backend name.
    # By omitting specific hardware 'triples', IREE generates 
    # portable SPIR-V that runs on any Vulkan-compliant GPU.
    export_output.compile(
        save_to=output_path, 
        target_backends=["vulkan-spirv"]
    )

    print(f"--- SUCCESS: {output_path} is ready for deployment ---")
    print("This binary is portable across AMD, NVIDIA, Intel, and Mobile Vulkan drivers.")

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    compile_for_vulkan()