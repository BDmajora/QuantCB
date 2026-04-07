import os
import torch
import torch.nn as nn
import torch._dynamo
from iree.turbine import aot 
from config import *
from models.quantcb_model import QuantCB_Model

# Enable autograd tracing for the backward pass
torch._dynamo.config.trace_autograd_ops = True

class VulkanTrainEngine(nn.Module):
    def __init__(self, model, lr=1e-4):
        super().__init__()
        # We store the model normally. 
        # Do NOT create a separate ParameterList here; it causes 'Aliasing' warnings.
        self.model = model
        self.lr = lr

    def forward(self, xb, yb):
        """
        Full Training Step: Forward -> Loss -> Backward -> Update
        """
        # 1. Forward Pass
        # Returns (logits, loss, aux_loss)
        logits, loss, _ = self.model(xb, yb)
        
        # 2. Reduction
        if loss.dim() > 0:
            loss = loss.mean()
        
        # 3. Capture Gradients
        # Accessing parameters directly inside the trace is the 'Strict' way.
        params = list(self.model.parameters())
        
        grads = torch.autograd.grad(
            loss, 
            params, 
            allow_unused=True,
            retain_graph=False
        )
        
        # 4. Optimizer Update
        self._apply_optimizer_update(params, grads)
        
        # FIX: .detach() prevents the 'autograd.grad consumed grad_fn' error
        return loss.detach()

    @torch.no_grad()
    def _apply_optimizer_update(self, params, grads):
        """Functional update for SPIR-V compatibility."""
        for p, g in zip(params, grads):
            if g is not None:
                # Use functional addition instead of in-place .add_() 
                # for better compatibility with strict-mode graph captures.
                p.copy_(p - (g * self.lr))

    def main(self, x):
        """Inference entry point."""
        logits, _, _ = self.model(x)
        return logits


def compile_training_binary():
    print("--- 2026 IREE-Turbine: Vulkan Training Bake Starting ---")
    
    model = QuantCB_Model(
        vocab_size=VOCAB_SIZE, 
        d_model=D_MODEL, 
        n_layers=N_LAYERS, 
        num_experts=NUM_EXPERTS, 
        top_k=TOP_K
    )
    
    engine = VulkanTrainEngine(model, lr=1e-4)
    
    # Inputs for RX 6800 (i64)
    example_xb = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, BLOCK_SIZE), dtype=torch.int64)
    example_yb = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, BLOCK_SIZE), dtype=torch.int64)
    
    print("Capturing Graph via aot.export...")
    
    try:
        # FIX: Removed 'strict=False' to fix TypeError.
        # The model is now clean enough to pass default 'strict' export.
        export_output = aot.export(engine, example_xb, example_yb)
    except Exception as e:
        print(f"\nEXPORT FAILED: {e}")
        raise e
    
    output_path = os.path.join(OUTPUT_DIR, "quantcb_vulkan_train.vmfb")
    print(f"Compiling to {output_path} for target: vulkan-spirv...")
    
    export_output.compile(
        save_to=output_path, 
        target_backends=["vulkan-spirv"]
    )

    print(f"--- SUCCESS: {output_path} is ready ---")

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    compile_training_binary()