import os
import torch
import time
from config import *
from lr_scheduler import get_lr
from data_engine import load_and_tag_all_data, get_batch
from models.quantcb_model import QuantCB_Model
from models.quantcb_engine import QuantCB_Engine
from tokenizer_basic import QuantCB_Tokenizer

def train():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    raw_text = load_and_tag_all_data()
    tokenizer = QuantCB_Tokenizer()
    tokenizer.train(raw_text, target_vocab_size=VOCAB_SIZE)
    
    train_data = torch.tensor(tokenizer.encode(raw_text), dtype=torch.long)
    print(f"Dataset Size: {len(train_data)} tokens")

    raw_model = QuantCB_Model(vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_layers=N_LAYERS, num_experts=NUM_EXPERTS, top_k=TOP_K)
    engine = QuantCB_Engine(raw_model).to(DEVICE)
    optimizer = torch.optim.AdamW(engine.parameters(), lr=MAX_LR, weight_decay=WEIGHT_DECAY)

    start_iter = 0
    if os.path.exists(CHECKPOINT_PATH):
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        raw_model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_iter = ckpt['iteration'] + 1

    print("--- Training Active. Ctrl+C to save final state. ---")
    try:
        t0 = time.time() # Start the timer for the first interval
        for iter in range(start_iter, ITERATIONS):
            lr = get_lr(iter, start_iter)
            for pg in optimizer.param_groups: pg['lr'] = lr

            xb, yb = get_batch(train_data, BATCH_SIZE, BLOCK_SIZE, DEVICE)
            logits, loss, _ = engine(xb, yb)
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(engine.parameters(), GRAD_CLIP)
            optimizer.step()

            # --- REPORTING (Every 10 Steps) ---
            if iter % 10 == 0:
                t1 = time.time()
                dt = t1 - t0 # Time elapsed for 10 steps
                # TPS Calculation: (Total Tokens in 10 steps) / (Seconds elapsed)
                tps = (BATCH_SIZE * BLOCK_SIZE * 10) / dt
                print(f"Step {iter:4d} | Loss: {loss.item():.4f} | LR: {lr:.2e} | TPS: {tps:.0f}")
                t0 = t1 # Reset timer for the next interval

            # --- HEALTH CHECK (Every 50 Steps) ---
            if iter % 50 == 0 and iter > 0:
                engine.eval()
                print(f"\n--- Health Check Step {iter} ---")
                with torch.no_grad():
                    # Generate seed using existing encode method
                    seed_str = f"{TAGS['truth']}{TAGS['stories']} "
                    context_ids = tokenizer.encode(seed_str)
                    context = torch.tensor([context_ids], dtype=torch.long, device=DEVICE)
                    
                    generated = engine.generate(context, max_new_tokens=40, temperature=0.8)
                    print(f"Output: {tokenizer.decode(generated[0].tolist())}\n")
                engine.train()

            # --- CHECKPOINTING (Every 50 Steps) ---
            if iter % 50 == 0:
                torch.save({'iteration': iter, 'model_state_dict': raw_model.state_dict(), 
                            'optimizer_state_dict': optimizer.state_dict()}, CHECKPOINT_PATH)

    except KeyboardInterrupt:
        print("\n--- Saving session state ---")
        torch.save({'iteration': iter, 'model_state_dict': raw_model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, CHECKPOINT_PATH)

if __name__ == "__main__":
    train()