import torch
import torch._dynamo
import os
from iree.turbine import aot 
from param_container import ParamContainer 

# Trace autograd.grad logic during export
torch._dynamo.config.trace_autograd_ops = True

from config import (
    VOCAB_SIZE, D_MODEL, N_LAYERS, N_HEADS, 
    HEAD_DIM, LATENT_DIM, NUM_EXPERTS, 
    BATCH_SIZE, SEQ_LENGTH, MAX_LR, OUTPUT_DIR
)
from models.quantcb_model import quantcb_model_forward_stateless

class TrainingModule(torch.nn.Module):
    """
    Module for training loop export. 
    Registers weights in ParameterDict to prevent constant lifting by torch.export.
    """
    def __init__(self, params: ParamContainer):
        super().__init__()
        self.params_dict = torch.nn.ParameterDict()
        for name, tensor in params.weights.items():
            # Replace dots with underscores for internal key compatibility
            safe_name = name.replace(".", "_DOT_")
            self.params_dict[safe_name] = torch.nn.Parameter(tensor.detach().clone())
        
        self.num_blocks = N_LAYERS
        
    def forward(self, x, y):
        # Reconstruct stateless dictionary from registered parameters
        stateless_weights = {
            k.replace("_DOT_", "."): v for k, v in self.params_dict.items()
        }
        
        _, loss, l_aux = quantcb_model_forward_stateless(
            idx=x,
            weights=stateless_weights, 
            num_blocks=self.num_blocks,
            targets=y,
            n_heads=N_HEADS,
            latent_dim=LATENT_DIM,
            head_dim=HEAD_DIM,
            num_experts=NUM_EXPERTS
        )
        
        if isinstance(l_aux, (list, tuple)):
            l_aux_scalar = sum(l_aux) if len(l_aux) > 0 else torch.tensor(0.0, device=x.device)
        else:
            l_aux_scalar = l_aux
            
        total_loss = (loss + l_aux_scalar).mean()
        
        # Access parameters directly from module state for grad calculation
        params_to_train = list(self.params_dict.values())
        
        grads = torch.autograd.grad(
            total_loss, 
            params_to_train, 
            allow_unused=True,
            create_graph=False
        )
        
        # Output list starts with detached loss for IREE compatibility
        results = [total_loss.detach()]
        for g, p in zip(grads, params_to_train):
            if g is None:
                results.append(torch.zeros_like(p))
            else:
                results.append(g.clone())
        
        return tuple(results)

def compile_for_vulkan():
    print(f"IREE Turbine: Baking {D_MODEL}d Training Kernel")
    
    param_container = ParamContainer(
        N_LAYERS, VOCAB_SIZE, D_MODEL, N_HEADS, 
        HEAD_DIM, LATENT_DIM, NUM_EXPERTS
    )
    
    module_to_export = TrainingModule(param_container)
    
    example_x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LENGTH), dtype=torch.int64)
    example_y = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LENGTH), dtype=torch.int64)

    print("Step 1: Tracing Graph via torch.export")
    try:
        # Exporting with explicit x and y arguments
        exported = aot.export(
            module_to_export, 
            args=(example_x, example_y)
        )
        
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

        output_path = os.path.join(OUTPUT_DIR, "quantcb_train_vulkan.vmfb")
        print(f"Step 2: Compiling to {output_path}")
        
        exported.compile(
            save_to=output_path, 
            target_backends=["vulkan-spirv"],
            flags=[
                "--iree-vulkan-target-env=rdna2-unknown-unknown", 
                "--iree-opt-const-eval=false",
                "--iree-hal-memoize-device-queries",
                "--iree-opt-strip-assertions=true"
            ] 
        )
        print("\nSUCCESS: GPU Training Kernel Ready")

    except Exception as e:
        print(f"\nFAILURE during export or compile: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    compile_for_vulkan()