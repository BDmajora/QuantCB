import time
import torch
import torch.nn.functional as F
from config import *
from lr_scheduler import get_lr
from data_engine import get_batch

class QuantCBTrainer:
    def __init__(self, engine, optimizer, train_data, tokenizer, start_iter, device):
        self.device = device 
        self.engine = engine
        self.optimizer = optimizer
        self.train_data = train_data
        self.tokenizer = tokenizer
        self.start_iter = start_iter
        
        # Pre-cache tag IDs to avoid string lookups in the hot loop
        self.hallucinate_id = self.tokenizer.encode(TAGS["hallucinate"], output_tensor=False)[0]

    def _save_checkpoint(self, iter_num):
        """Standard CPU-side save."""
        # Move state to CPU momentarily for the save
        torch.save({
            'iteration': iter_num, 
            'model_state_dict': self.engine.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, CHECKPOINT_PATH)

    def _health_check(self, iter_num):
        """Inference on Vulkan Device."""
        self.engine.eval()
        print(f"\n--- Vulkan Health Check Step {iter_num} ---")
        with torch.no_grad():
            seed_str = f"{TAGS['truth']}{TAGS['stories']} "
            # Encode and move to Vulkan
            context_ids = self.tokenizer.encode(seed_str)
            context = context_ids.unsqueeze(0).to(self.device)
            
            # Pure Vulkan Generation
            generated = self.engine.generate(context, max_new_tokens=40)
            print(f"Output: {self.tokenizer.decode(generated[0])}\n")
        self.engine.train()

    def train_step(self, iter_num):
        """Asynchronous Dispatch to RX 6800."""
        # 1. Update LR on CPU Traffic Controller
        lr = get_lr(iter_num, self.start_iter)
        for pg in self.optimizer.param_groups: 
            pg['lr'] = lr

        # 2. Fetch Pinned Batch and Ship to Vulkan
        # non_blocking=True is the key to the Vulkan Timeline
        xb, yb = get_batch(self.train_data, BATCH_SIZE, BLOCK_SIZE, "cpu")
        xb, yb = xb.pin_memory().to(self.device, non_blocking=True), yb.pin_memory().to(self.device, non_blocking=True)
        
        # 3. Vulkan Logic: Compute Drift Targets on-device
        is_corrupted = (xb == self.hallucinate_id).any(dim=1, keepdim=True)
        drift_targets = is_corrupted.float().expand(-1, BLOCK_SIZE).contiguous().to(self.device)
        
        # 4. Vulkan Forward/Backward Pass
        self.optimizer.zero_grad(set_to_none=True)
        
        # Execution starts on RX 6800 here
        logits, loss, probe_logits_list = self.engine(xb, yb)
        
        # Multi-Loop Probe Aggregation (BCE on Vulkan)
        probe_logits = torch.stack(probe_logits_list).mean(dim=0)
        bce_loss = F.binary_cross_entropy_with_logits(
            probe_logits.view(-1), 
            drift_targets.view(-1)
        )
        
        total_loss = loss + (0.5 * bce_loss)
        total_loss.backward()
        
        # 5. Gradient Clipping & Optimizer Step (GPU Kernels)
        torch.nn.utils.clip_grad_norm_(self.engine.parameters(), GRAD_CLIP)
        self.optimizer.step()

        # We only return scalars. .item() causes a sync-point; 
        # this is the only time the CPU waits for the GPU.
        return total_loss.item(), bce_loss.item(), lr

    def run(self):
        """High-Throughput Training Loop."""
        print(f"--- RX 6800 Vulkan Timeline Active ---")
        t0 = time.time()
        
        try:
            for iter_num in range(self.start_iter, ITERATIONS):
                loss, bce_loss, lr = self.train_step(iter_num)

                # Throughput monitoring every 10 steps
                if iter_num % 10 == 0:
                    t1 = time.time()
                    # Calculate TPS based on actual GPU retirement speed
                    tps = (BATCH_SIZE * BLOCK_SIZE * 10) / (t1 - t0)
                    print(f"Step {iter_num:4d} | Loss: {loss:.4f} | LR: {lr:.2e} | TPS: {tps:.0f}")
                    t0 = t1

                if iter_num % 50 == 0 and iter_num > 0:
                    self._health_check(iter_num)
                    self._save_checkpoint(iter_num)

        except KeyboardInterrupt:
            print("\n--- Interrupted. Final Save... ---")
            self._save_checkpoint(iter_num)