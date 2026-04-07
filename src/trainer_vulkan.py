import time
import numpy as np
import iree.runtime as ireert
from config import *
from data_engine import get_batch

class IREEVulkanTrainer:
    def __init__(self, config, vmfb_path, train_data, tokenizer, start_iter):
        self.config = config
        self.train_data = train_data
        self.tokenizer = tokenizer
        self.start_iter = start_iter
        
        # Initialize Vulkan Session
        print(f"Binding RX 6800 Vulkan Context...")
        self.vm_module = ireert.VmModule.mmap(self.config.vm_instance, str(vmfb_path))
        self.ctx = ireert.SystemContext(config=self.config)
        self.ctx.add_vm_module(self.vm_module)
        
        # Access the compiled training logic
        self.bound_module = self.ctx.modules.module

    def _save_weights(self, iter_num):
        """
        Pulls weights back from GPU memory to save them to disk.
        """
        print(f"Checkpointing weights at Step {iter_num}...")
        try:
            # Fetches the tuple of tensors we exposed in the PyTorch engine
            weight_tensors = self.bound_module.export_weights()
            save_path = f"{OUTPUT_DIR}/weights_step_{iter_num}.npz"
            
            # Convert DeviceArrays to numpy arrays for standard saving
            weights_np = {f"layer_{i}": np.array(w) for i, w in enumerate(weight_tensors)}
            np.savez(save_path, **weights_np)
            
        except AttributeError:
            print("WARNING: 'export_weights' function not found in VMFB.")

    def _health_check(self, iter_num):
        """Direct-on-GPU validation."""
        print(f"\n--- Vulkan Health Check: Step {iter_num} ---")
        test_input = np.random.randint(0, VOCAB_SIZE, (1, BLOCK_SIZE), dtype=np.int64)
        
        # Call inference function inside the training module
        results = self.bound_module.main(test_input)
        print(f"Inference Live. Output Logits Shape: {results[0].shape}\n")

    def train_step(self, iter_num):
        """
        The Heavy Lift: One call to the GPU for a full update cycle.
        """
        # 1. Fetch Batch (CPU)
        xb_torch, yb_torch = get_batch(self.train_data, BATCH_SIZE, BLOCK_SIZE, "cpu")
        xb = xb_torch.numpy().astype(np.int64)
        yb = yb_torch.numpy().astype(np.int64)

        # 2. SPIR-V Dispatch (Forward -> Loss -> Grad -> Optimizer Update)
        results = self.bound_module.train_step(xb, yb)
        
        # Convert DeviceArray back to Python float
        loss_val = float(results[0])
        return loss_val

    def run(self):
        """The Performance Loop."""
        print(f"--- Vulkan Timeline Active on RX 6800 ---")
        t0 = time.time()
        
        try:
            for iter_num in range(self.start_iter, ITERATIONS):
                # EXECUTE GPU KERNEL
                loss = self.train_step(iter_num)

                # Monitor performance
                if iter_num % 10 == 0:
                    t1 = time.time()
                    tps = (BATCH_SIZE * BLOCK_SIZE * 10) / (t1 - t0)
                    print(f"Step {iter_num:4d} | Loss: {loss:.4f} | TPS: {tps:.0f}")
                    t0 = t1

                if iter_num % 50 == 0 and iter_num > 0:
                    self._health_check(iter_num)
                    self._save_weights(iter_num)

        except KeyboardInterrupt:
            print("\n--- Training Stopped by User. ---")