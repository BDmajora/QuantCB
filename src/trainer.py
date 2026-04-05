import os
import time
import torch
import torch.nn.functional as F
from config import *
from lr_scheduler import get_lr
from data_engine import get_batch

class QuantCBTrainer:
    def __init__(self, engine, optimizer, train_data, tokenizer, start_iter, hallucinate_id):
        self.device = DEVICE 
        self.engine = engine
        self.optimizer = optimizer
        self.train_data = train_data
        self.tokenizer = tokenizer
        self.start_iter = start_iter
        self.hallucinate_id = hallucinate_id

    def _save_checkpoint(self, iter_num):
        """Saves the model and optimizer states efficiently."""
        torch.save({
            'iteration': iter_num, 
            'model_state_dict': self.engine.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, CHECKPOINT_PATH)

    def _health_check(self, iter_num):
        """Generates a text sample using optimized inference."""
        self.engine.eval()
        print(f"\n--- Health Check Step {iter_num} ---")
        with torch.no_grad():
            seed_str = f"{TAGS['truth']}{TAGS['stories']} "
            context_ids = self.tokenizer.encode(seed_str)
            context = torch.tensor([context_ids], dtype=torch.long, device=self.device)
            
            # FIXED: Disabled BF16 Autocast on CPU to prevent thread deadlocks.
            # On most CPUs, FP32 is actually faster than emulated BF16.
            with torch.autocast(device_type='cpu', enabled=False):
                generated = self.engine.generate(context, max_new_tokens=40)
            
            print(f"Output: {self.tokenizer.decode(generated[0].tolist())}\n")
        self.engine.train()

    def train_step(self, iter_num):
        """Executes a single forward and backward pass with zero unnecessary overhead."""
        lr = get_lr(iter_num, self.start_iter)
        for pg in self.optimizer.param_groups: 
            pg['lr'] = lr

        # 1. Fetch vectorized batch
        xb, yb = get_batch(self.train_data, BATCH_SIZE, BLOCK_SIZE, self.device)
        
        # 2. Memory-efficient drift targeting
        is_corrupted = (xb == self.hallucinate_id).any(dim=1, keepdim=True)
        # Using .contiguous() here ensures the expansion doesn't cause issues during the BCE loss calc
        drift_targets = is_corrupted.float().expand(-1, BLOCK_SIZE).contiguous()
        
        # 3. CPU Forward Pass
        # FIXED: Removed 'dtype=torch.bfloat16'. 
        # Forcing BF16 on CPU without AVX-512_BF16 support causes massive hangs.
        with torch.autocast(device_type='cpu', enabled=False):
            logits, loss, probe_logits_list = self.engine(xb, yb)
            
            # Aggregate probes from all loops
            probe_logits = torch.stack(probe_logits_list).mean(dim=0)
            
            # Using binary_cross_entropy_with_logits is numerically stable for FP32
            bce_loss = F.binary_cross_entropy_with_logits(
                probe_logits.reshape(-1), 
                drift_targets.reshape(-1)
            )
            
            total_loss = loss + (0.5 * bce_loss)
        
        # 4. Optimizer Step
        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        
        # Gradient clipping is vital when using higher learning rates in MTP/MoE setups
        torch.nn.utils.clip_grad_norm_(self.engine.parameters(), GRAD_CLIP)
        self.optimizer.step()

        return total_loss.item(), bce_loss.item(), lr

    def run(self):
        """Main training loop with real-time TPS monitoring."""
        print(f"--- Phase 2 Active: {MAX_LOOPS} Loops | {EXIT_THRESHOLD} Entropy Exit ---")
        t0 = time.time()
        
        try:
            for iter_num in range(self.start_iter, ITERATIONS):
                loss, bce_loss, lr = self.train_step(iter_num)

                if iter_num % 10 == 0:
                    t1 = time.time()
                    dt = t1 - t0 
                    # Correct TPS calculation for throughput monitoring
                    tps = (BATCH_SIZE * BLOCK_SIZE * 10) / dt
                    print(f"Step {iter_num:4d} | Loss: {loss:.4f} (Probe: {bce_loss:.4f}) | LR: {lr:.2e} | TPS: {tps:.0f}")
                    t0 = t1

                if iter_num % 50 == 0:
                    if iter_num > 0: 
                        self._health_check(iter_num)
                    self._save_checkpoint(iter_num)

        except KeyboardInterrupt:
            print("\n--- Saving session state before exit ---")
            self._save_checkpoint(iter_num)