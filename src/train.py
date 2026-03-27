import os
import torch
import torch.optim as optim
import math
import sys
from datasets import load_dataset # The new reliable way
from models.quantcb_model import QuantCB_Model
from models.quantcb_engine import QuantCB_Engine
from tokenizer_basic import QuantCB_Tokenizer

# --- SETTINGS ---
ITERATIONS = 500      
BATCH_SIZE = 16       
BLOCK_SIZE = 64       
MAX_LR = 5e-4
WARMUP_STEPS = 50
FORCE_REBUILD = True  

TAGS = {
    "shakespeare": "<|shkspeare|>",
    "tiny_stories": "<|story|>",
    "tiny_wiki": "<|wiki|>"
}

def get_batch(data, batch_size, block_size):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

def train():
    device = 'cpu'
    torch.set_num_threads(4)
    
    output_dir = "modelOutput"
    os.makedirs(output_dir, exist_ok=True)

    combined_raw_text = ""
    print("--- LOADING DATA VIA HUGGINGFACE DATASETS ---")

    # 1. Load TinyStories (Official)
    try:
        print("Fetching TinyStories...")
        ds_stories = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
        # Take a small slice for the 500-step run
        for i, ex in enumerate(ds_stories.take(1000)):
            combined_raw_text += f"{TAGS['tiny_stories']}\n{ex['text']}\n"
        print(f"SUCCESS: Integrated TinyStories")
    except Exception as e:
        print(f"FAILED TinyStories: {e}")

    # 2. Load Wiki (Official Simple Wikipedia)
    try:
        print("Fetching Wiki...")
        ds_wiki = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
        for i, ex in enumerate(ds_wiki.take(500)):
            combined_raw_text += f"{TAGS['tiny_wiki']}\n{ex['text'][:2000]}\n"
        print(f"SUCCESS: Integrated Wiki")
    except Exception as e:
        print(f"FAILED Wiki: {e}")

    # 3. Load Shakespeare (Still reliable via raw URL)
    try:
        import requests
        print("Fetching Shakespeare...")
        r = requests.get("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")
        combined_raw_text += f"{TAGS['shakespeare']}\n{r.text[:500000]}\n"
        print(f"SUCCESS: Integrated Shakespeare")
    except Exception as e:
        print(f"FAILED Shakespeare: {e}")

    if len(combined_raw_text) < 1000:
        print("CRITICAL: Combined text is too small. Check internet connection.")
        sys.exit(1)

    # --- TOKENIZER & MODEL (Same as before) ---
    vocab_size = 2048 
    tokenizer = QuantCB_Tokenizer()
    tok_path = os.path.join(output_dir, "tokenizer.json")
    
    if os.path.exists(tok_path): os.remove(tok_path)
    
    print("\nTraining Tokenizer...")
    tokenizer.train(combined_raw_text, target_vocab_size=vocab_size)
    tokenizer.save(tok_path)

    encoded_data = tokenizer.encode(combined_raw_text)
    data_tensor = torch.tensor(encoded_data, dtype=torch.long)
    train_data = data_tensor[:int(0.9*len(data_tensor))]

    raw_model = QuantCB_Model(vocab_size=vocab_size, d_model=256, n_layers=4, num_experts=8)
    engine = QuantCB_Engine(raw_model).to(device)
    optimizer = optim.AdamW(engine.parameters(), lr=MAX_LR)

    print(f"\n--- TRAINING {ITERATIONS} STEPS ---")
    for iter in range(ITERATIONS):
        xb, yb = get_batch(train_data, BATCH_SIZE, BLOCK_SIZE)
        _, loss, _ = engine(xb, yb)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iter % 50 == 0:
            print(f"Step {iter} | Loss: {loss.item():.4f}")

    torch.save(raw_model.state_dict(), os.path.join(output_dir, 'quantcb_final.pth'))
    print("Done.")

if __name__ == "__main__":
    train()