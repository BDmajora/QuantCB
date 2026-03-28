import os
import torch
import torch.nn.functional as F
import time
from config import *
from lr_scheduler import get_lr
from data_engine import load_and_tag_all_data, get_batch
from models.quantcb_model import QuantCB_Model
from models.ouro_engine import Ouro_Engine 
from tokenizer_basic import QuantCB_Tokenizer

def train():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    raw_text = load_and_tag_all_data()
    tokenizer = QuantCB_Tokenizer()
    tokenizer.train(raw_text, target_vocab_size=VOCAB_SIZE)
    
    train_data = torch.tensor(tokenizer.encode(raw_text), dtype=torch.long)
    print(f"Dataset Size: {len(train_data)} tokens")

    # Tag for supervision
    hallucinate_id = tokenizer.encode(TAGS["hallucinate"])[0]

    raw_model = QuantCB_Model(
        vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_layers=N_LAYERS, 
        num_experts=NUM_EXPERTS, top_k=TOP_K
    )
    
    engine = Ouro_Engine(raw_model, max_loops=MAX_LOOPS, exit_threshold=EXIT_THRESHOLD).to(DEVICE)
    optimizer = torch.optim.AdamW(engine.parameters(), lr=MAX_LR, weight_decay=WEIGHT_DECAY)

    start_iter = 0
    if os.path.exists(CHECKPOINT_PATH):
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        engine.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_iter = ckpt['iteration'] + 1

    print(f"--- Phase 2 Active: {MAX_LOOPS} Loops | {EXIT_THRESHOLD} Entropy Exit ---")
    
    try:
        t0 = time.time()
        for iter in range(start_iter, ITERATIONS):
            lr = get_lr(iter, start_iter)
            for pg in optimizer.param_groups: pg['lr'] = lr

            # 1. Get Batch (Unpack 2 values)
            xb, yb = get_batch(train_data, BATCH_SIZE, BLOCK_SIZE, DEVICE)
            
            # 2. Identify Corrupted Sequences
            is_corrupted = (xb == hallucinate_id).any(dim=1, keepdim=True)
            drift_targets = is_corrupted.expand_as(xb).float().to(DEVICE)
            
            # 3. Forward pass (returns probe_logits as a list from loops)
            logits, loss, probe_logits_list = engine(xb, yb)
            
            # 4. FIX: Process probe list into a single tensor
            # We average the probe predictions across all recursion loops
            probe_logits = torch.stack(probe_logits_list).mean(dim=0)
            
            # 5. Calculate BCE loss for the latent_probe
            bce_loss = F.binary_cross_entropy_with_logits(
                probe_logits.view(-1), 
                drift_targets.view(-1)
            )
            
            total_loss = loss + (0.5 * bce_loss)
            
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(engine.parameters(), GRAD_CLIP)
            optimizer.step()

            if iter % 10 == 0:
                t1 = time.time()
                dt = t1 - t0 
                tps = (BATCH_SIZE * BLOCK_SIZE * 10) / dt
                print(f"Step {iter:4d} | Loss: {total_loss.item():.4f} (Probe: {bce_loss.item():.4f}) | LR: {lr:.2e} | TPS: {tps:.0f}")
                t0 = t1

            if iter % 50 == 0 and iter > 0:
                engine.eval()
                print(f"\n--- Health Check Step {iter} ---")
                with torch.no_grad():
                    seed_str = f"{TAGS['truth']}{TAGS['stories']} "
                    context = torch.tensor([tokenizer.encode(seed_str)], dtype=torch.long, device=DEVICE)
                    generated = engine.generate(context, max_new_tokens=40)
                    print(f"Output: {tokenizer.decode(generated[0].tolist())}\n")
                engine.train()

            if iter % 50 == 0:
                torch.save({
                    'iteration': iter, 
                    'model_state_dict': engine.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, CHECKPOINT_PATH)

    except KeyboardInterrupt:
        print("\n--- Saving session state ---")
        torch.save({'iteration': iter, 'model_state_dict': engine.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, CHECKPOINT_PATH)

if __name__ == "__main__":
    train()