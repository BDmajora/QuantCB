import os
import torch
import torch.nn as nn
import torch.optim as optim
import math
import time
import sys
from datasets import load_dataset
from models.quantcb_model import QuantCB_Model
from models.quantcb_engine import QuantCB_Engine
from tokenizer_basic import QuantCB_Tokenizer

# --- HIGH MEMORY / 500 STEP SETTINGS ---
ITERATIONS = 500      
BATCH_SIZE = 64        
BLOCK_SIZE = 256       
MAX_LR = 3e-4          
GRAD_CLIP = 1.0        
WARMUP_STEPS = 50      
WEIGHT_DECAY = 0.1     

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

def get_lr(step):
    if step < WARMUP_STEPS:
        return MAX_LR * step / WARMUP_STEPS
    decay_ratio = (step - WARMUP_STEPS) / (ITERATIONS - WARMUP_STEPS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * max(0, min(1, decay_ratio))))
    return 0.1 * MAX_LR + coeff * 0.9 * MAX_LR

def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_dir = "modelOutput"
    checkpoint_path = os.path.join(output_dir, "checkpoint.pth")
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

    # 2. TOKENIZER & DATA PREP
    vocab_size = 2048 
    tokenizer = QuantCB_Tokenizer()
    tok_path = os.path.join(output_dir, "tokenizer.json")
    tokenizer.train(combined_raw_text, target_vocab_size=vocab_size)
    tokenizer.save(tok_path)

    encoded_data = tokenizer.encode(combined_raw_text)
    train_data = torch.tensor(encoded_data, dtype=torch.long)
    print(f"Dataset Size: {len(train_data)} tokens")

    # 3. MODEL INIT
    raw_model = QuantCB_Model(
        vocab_size=vocab_size, 
        d_model=384,     
        n_layers=6, 
        num_experts=8, 
        top_k=2
    )
    engine = QuantCB_Engine(raw_model).to(device)
    optimizer = optim.AdamW(engine.parameters(), lr=MAX_LR, weight_decay=WEIGHT_DECAY)

    # --- RESUME LOGIC ---
    start_iter = 0
    if os.path.exists(checkpoint_path):
        print(f"--- Found Checkpoint! Resuming from disk ---")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        raw_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_iter = checkpoint['iteration'] + 1
        print(f"Resuming at Step {start_iter}")

    print(f"\n--- STARTING 500 STEP SPRINT ---")
    t0 = time.time()

    for iter in range(start_iter, ITERATIONS):
        lr = get_lr(iter)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        xb, yb = get_batch(train_data, BATCH_SIZE, BLOCK_SIZE, device)
        
        # Forward & Loss
        logits, loss, _ = engine(xb, yb)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(engine.parameters(), GRAD_CLIP)
        optimizer.step()

        # Reporting Frequency (Every 5 steps)
        if iter % 5 == 0:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1 
            tokens_processed = BATCH_SIZE * BLOCK_SIZE * 5 if iter > start_iter else BATCH_SIZE * BLOCK_SIZE
            tps = tokens_processed / dt
            print(f"Step {iter:3d} | Loss: {loss.item():.4f} | LR: {lr:.2e} | TPS: {tps:.0f}")

        # SAVE CHECKPOINT every 50 steps
        if iter % 50 == 0 and iter > 0:
            torch.save({
                'iteration': iter,
                'model_state_dict': raw_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, checkpoint_path)
            print(f"--- Checkpoint Saved at Step {iter} ---")

        # Health Check Frequency (Every 25 steps)
        if iter % 25 == 0 and iter > 0:
            engine.eval()
            print(f"\n--- Health Check Step {iter} ---")
            with torch.no_grad():
                context = torch.zeros((1, 1), dtype=torch.long, device=device) 
                generated = engine.generate(context, max_new_tokens=40, temperature=0.8)
                decoded = tokenizer.decode(generated[0].tolist())
                print(f"Output: {decoded}\n")
            engine.train()

    # Final Save
    torch.save(raw_model.state_dict(), os.path.join(output_dir, 'quantcb_final.pth'))
    print("Done. Model saved.")

if __name__ == "__main__":
    train()