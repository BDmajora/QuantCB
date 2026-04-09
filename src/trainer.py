import time
import numpy as np
import torch
import iree.runtime as ireert
from config import *
from lr_scheduler import get_lr
from data_engine import get_batch

class QuantCBTrainer:
    def __init__(self, engine, config, train_data, tokenizer, start_iter):
        self.engine = engine          # The compiled IREE module/engine
        self.config = config          # IREE device config (Vulkan)
        self.train_data = train_data
        self.tokenizer = tokenizer
        self.start_iter = start_iter
        
        # Pre-encode the tag for performance inside the training loop
        tag_tokens = self.tokenizer.encode(TAGS["hallucinate"])
        self.hallucinate_id = tag_tokens[0] if tag_tokens else -1

    def _save_checkpoint(self, iter_num):
        """Metadata checkpointing."""
        # Ensure the directory exists (defined in config.py)
        torch.save({
            'iteration': iter_num, 
        }, CHECKPOINT_PATH)
        # Note: In a full AOT setup, weights are usually managed by 
        # IREE's Parameter Manager or extracted via engine.get_state()

    def _health_check(self, iter_num):
        """Inference using the compiled IREE module."""
        print(f"\n--- Vulkan Health Check Step {iter_num} ---")
        seed_str = f"{TAGS['truth']}{TAGS['stories']} "
        
        # 1. Encode context into a NumPy array
        context_list = self.tokenizer.encode(seed_str)
        context_ids = np.array([context_list], dtype=np.int32)
        
        # 2. Map to Vulkan Device
        v_input = ireert.asdevicearray(self.config.device, context_ids)
        
        # 3. Dispatch to the 'generate' function in your compiled module
        # Note: IREE returns DeviceArrays; we convert them back to Host NumPy
        generated = self.engine.generate(v_input)
        
        out_np = np.asarray(generated)
        print(f"Output: {self.tokenizer.decode(out_np[0].tolist())}\n")

    def train_step(self, iter_num):
        """Asynchronous Dispatch to the Vulkan Driver."""
        # 1. Update LR
        lr = get_lr(iter_num, self.start_iter)

        # 2. Fetch NumPy Batch (No .numpy() needed, get_batch returns ndarray)
        xb_np, yb_np = get_batch(self.train_data, BATCH_SIZE, BLOCK_SIZE)
        
        # 3. Compute Drift Targets (Host-side prep)
        # This checks which sequences contain the 'hallucinate' tag
        is_corrupted = (xb_np == self.hallucinate_id).any(axis=1, keepdims=True)
        drift_targets = np.broadcast_to(is_corrupted, (BATCH_SIZE, BLOCK_SIZE)).astype(np.float32)
        
        # 4. Map buffers to Vulkan Device
        # Using asdevicearray ensures zero-copy if the hardware supports Unified Memory
        v_xb = ireert.asdevicearray(self.config.device, xb_np)
        v_yb = ireert.asdevicearray(self.config.device, yb_np)
        v_drift = ireert.asdevicearray(self.config.device, drift_targets)
        v_lr = ireert.asdevicearray(self.config.device, np.array(lr, dtype=np.float32))

        # 5. Execute Compiled Training Step on Vulkan
        # results will usually be a list/tuple of DeviceArrays
        results = self.engine.train_step(v_xb, v_yb, v_drift, v_lr)

        # 6. Retrieve scalars from Device to Host
        # Assuming train_step returns [total_loss, bce_loss]
        total_loss = np.asarray(results[0]).item()
        bce_loss = np.asarray(results[1]).item()
        
        return total_loss, bce_loss, lr

    def run(self):
        """High-Throughput Training Loop."""
        print(f"--- IREE-Turbine Vulkan Timeline Active ---")
        t0 = time.time()
        
        try:
            for iter_num in range(self.start_iter, ITERATIONS):
                loss, bce_loss, lr = self.train_step(iter_num)

                # Throughput monitoring
                if iter_num % 10 == 0:
                    t1 = time.time()
                    dt = t1 - t0
                    # Tokens Per Second calculation
                    tps = (BATCH_SIZE * BLOCK_SIZE * 10) / dt if dt > 0 else 0
                    print(f"Step {iter_num:4d} | Loss: {loss:.4f} | LR: {lr:.2e} | TPS: {tps:.0f}")
                    t0 = t1

                if iter_num % 50 == 0 and iter_num > 0:
                    self._health_check(iter_num)
                    self._save_checkpoint(iter_num)

        except KeyboardInterrupt:
            print("\n--- Interrupted. Final Save... ---")
            # Using the last known iter_num
            self._save_checkpoint(locals().get('iter_num', self.start_iter))