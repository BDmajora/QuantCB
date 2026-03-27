import os
import torch
import torch.nn as nn
import torch.optim as optim
import math
import time
from datasets import load_dataset
from models.quantcb_model import QuantCB_Model
from models.quantcb_engine import QuantCB_Engine
from tokenizer_basic import QuantCB_Tokenizer

# --- CONFIGURATION ---
ITERATIONS = 6000      
BATCH_SIZE = 64        
BLOCK_SIZE = 256       
MAX_LR = 3e-4          
GRAD_CLIP = 1.0        
WARMUP_STEPS = 50      
RESUME_WARMUP = 50      # Added: Steps to settle optimizer after a restart
WEIGHT_DECAY = 0.1     

# --- ANTI-DEMENTIA / EARLY EXIT SETTINGS ---
LOSS_TARGET = 1.50      
WINDOW_SIZE = 10        
recent_losses = []      

TAGS = {
    "shakespeare": "<|shkspeare|>",
    "tiny_stories": "<|story|>",
    "tiny_wiki": "<|wiki|>"
}

def get_batch(data, batch_size, block_size, device):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

def get_lr(step, start_iter):
    # Logic for a fresh start warmup
    if step < WARMUP_STEPS:
        return MAX_LR * step / WARMUP_STEPS
    
    # Logic for Resume Warmup (Prevents Optimizer Shock)
    # Scales LR from 10% back to target over RESUME_WARMUP steps
    if start_iter > 0 and step < (start_iter + RESUME_WARMUP):
        progress = (step - start_iter) / RESUME_WARMUP
        target_lr = get_lr_base(step)
        return target_lr * (0.1 + 0.9 * progress)

    return get_lr_base(step)

def get_lr_base(step):
    decay_ratio = (step - WARMUP_STEPS) / (ITERATIONS - WARMUP_STEPS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * max(0, min(1, decay_ratio))))
    return 0.1 * MAX_LR + coeff * 0.9 * MAX_LR

def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_dir = "modelOutput"
    checkpoint_main = os.path.join(output_dir, "checkpoint.pth")
    os.makedirs(output_dir, exist_ok=True)

    # 1. DATA LOADING
    combined_raw_text = ""
    print("--- LOADING DATA INTO RAM ---")
    try:
        ds_stories = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
        for i, ex in enumerate(ds_stories.take(5000)):
            combined_raw_text += f"{TAGS['tiny_stories']} {ex['text']}\n"
        
        ds_wiki = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
        for i, ex in enumerate(ds_wiki.take(2000)):
            combined_raw_text += f"{TAGS['tiny_wiki']} {ex['text'][:1500]}\n"
    except Exception as e:
        print(f"Data Fetch Error: {e}")

    # 2. TOKENIZER
    vocab_size = 2048 
    tokenizer = QuantCB_Tokenizer()
    tok_path = os.path.join(output_dir, "tokenizer.json")
    tokenizer.train(combined_raw_text, target_vocab_size=vocab_size)
    tokenizer.save(tok_path)

    encoded_data = tokenizer.encode(combined_raw_text)
    train_data = torch.tensor(encoded_data, dtype=torch.long)
    print(f"Dataset Size: {len(train_data)} tokens")

    # 3. MODEL INIT
    raw_model = QuantCB_Model(vocab_size=vocab_size, d_model=384, n_layers=6, num_experts=8, top_k=2)
    engine = QuantCB_Engine(raw_model).to(device)
    optimizer = optim.AdamW(engine.parameters(), lr=MAX_LR, weight_decay=WEIGHT_DECAY)

    # --- RESUME LOGIC ---
    start_iter = 0
    if os.path.exists(checkpoint_main):
        print(f"--- Found Checkpoint! Resuming with Warmup ---")
        checkpoint = torch.load(checkpoint_main, map_location=device)
        raw_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_iter = checkpoint['iteration'] + 1
        print(f"Resuming at Step {start_iter}")

    print(f"\n--- SPRINTING TO 6000 (WITH {LOSS_TARGET} AUTO-EXIT) ---")
    t0 = time.time()

    for iter in range(start_iter, ITERATIONS):
        # Pass start_iter to handle resume warmup logic
        lr = get_lr(iter, start_iter)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        xb, yb = get_batch(train_data, BATCH_SIZE, BLOCK_SIZE, device)
        logits, loss, _ = engine(xb, yb)
        current_loss = loss.item()
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(engine.parameters(), GRAD_CLIP)
        optimizer.step()

        # Reporting & Exit Logic
        if iter % 5 == 0:
            recent_losses.append(current_loss)
            if len(recent_losses) > WINDOW_SIZE:
                recent_losses.pop(0)
            
            avg_loss = sum(recent_losses) / len(recent_losses)
            t1 = time.time()
            dt = t1 - t0
            t0 = t1 
            tps = (BATCH_SIZE * BLOCK_SIZE * 5) / dt
            print(f"Step {iter:3d} | Loss: {current_loss:.4f} | Avg: {avg_loss:.4f} | LR: {lr:.2e} | TPS: {tps:.0f}")

            if len(recent_losses) == WINDOW_SIZE and avg_loss <= LOSS_TARGET:
                print(f"\n--- TARGET STABILIZED: Avg Loss {avg_loss:.4f} <= {LOSS_TARGET} ---")
                break

        # ROLLING BACKUPS
        if iter % 50 == 0 and iter > 0:
            ckpt_name = f"ckpt_step_{iter}_loss_{current_loss:.2f}.pth"
            state = {
                'iteration': iter,
                'model_state_dict': raw_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': current_loss,
            }
            torch.save(state, os.path.join(output_dir, ckpt_name))
            torch.save(state, checkpoint_main)
            print(f"--- Snapshot Locked: {ckpt_name} ---")

        # Health Check
        if iter % 25 == 0 and iter > 0:
            engine.eval()
            print(f"\n--- Health Check Step {iter} ---")
            with torch.no_grad():
                context = torch.zeros((1, 1), dtype=torch.long, device=device) 
                generated = engine.generate(context, max_new_tokens=40, temperature=0.8)
                print(f"Output: {tokenizer.decode(generated[0].tolist())}\n")
            engine.train()

    # Final Export
    torch.save(raw_model.state_dict(), os.path.join(output_dir, 'QuantCB1.0_Final.pth'))
    print("Done. QuantCB 1.0 is officially ready.")

if __name__ == "__main__":
    train()