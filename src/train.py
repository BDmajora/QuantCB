import os
import urllib.request
import torch
import torch.optim as optim
from models.quantcb_model import QuantCB_Model
from tokenizer_basic import QuantCB_Tokenizer

def get_batch(data, batch_size, block_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # --- Pathing ---
    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(src_dir)
    output_dir = os.path.join(project_root, "modelOutput")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Data Loading
    data_path = os.path.join(output_dir, "input.txt")
    if not os.path.exists(data_path):
        print("Downloading TinyShakespeare...")
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        urllib.request.urlretrieve(url, data_path)
    
    with open(data_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    # 2. Tokenizer (Fixed Vocab)
    vocab_size = 2048 
    tokenizer = QuantCB_Tokenizer()
    tok_path = os.path.join(output_dir, "tokenizer.json")

    if os.path.exists(tok_path):
        tokenizer.load(tok_path)
    else:
        tokenizer.train(raw_text, target_vocab_size=vocab_size)
        tokenizer.save(tok_path)

    # 3. Encoding
    encoded_data = tokenizer.encode(raw_text)
    data_tensor = torch.tensor(encoded_data, dtype=torch.long)

    # 4. Model Setup (The Fair Benchmark)
    # d_ff=512 * top_k=2 == 1024 active compute units
    model = QuantCB_Model(
        vocab_size=vocab_size, 
        d_model=256, 
        n_heads=8, 
        d_ff=512,        
        n_layers=4,
        latent_dim=128, 
        head_dim=64,
        num_experts=8,
        top_k=2
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    print(f"Model initialized on {device}.")
    print(f"Benchmark: MLA + MoE (8 Experts, 2 Active, {vocab_size} Vocab)")

    # 5. Training Loop
    model.train()
    for iter in range(501): 
        xb, yb = get_batch(data_tensor, 32, 128, device)
        
        logits, loss, _ = model(xb, yb) 
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if iter % 20 == 0:
            print(f"Step {iter}: Loss {loss.item():.4f}")

    # Save
    save_path = os.path.join(output_dir, 'quantcb_base.pth')
    torch.save(model.state_dict(), save_path)
    print(f"Training complete. Saved to {save_path}")

if __name__ == "__main__":
    train()