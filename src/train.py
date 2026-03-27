import os
import urllib.request
import torch
import torch.optim as optim
import torch.nn as nn
from models.quantcb_model import QuantCB_Model
from models.quantcb_engine import QuantCB_Engine
from tokenizer_basic import QuantCB_Tokenizer

def get_batch(data, batch_size, block_size, device):
    # We need block_size + 1 to provide the targets for n+2 prediction
    # ix picks a starting point that leaves room for x (block_size) and y (shifted)
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Pathing Setup
    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(src_dir)
    output_dir = os.path.join(project_root, "modelOutput")
    os.makedirs(output_dir, exist_ok=True)

    # Data Acquisition
    data_path = os.path.join(output_dir, "input.txt")
    if not os.path.exists(data_path):
        print("Downloading TinyShakespeare...")
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        urllib.request.urlretrieve(url, data_path)
    
    with open(data_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    # Tokenizer Sync
    vocab_size = 2048 
    tokenizer = QuantCB_Tokenizer()
    tok_path = os.path.join(output_dir, "tokenizer.json")

    if os.path.exists(tok_path):
        tokenizer.load(tok_path)
    else:
        tokenizer.train(raw_text, target_vocab_size=vocab_size)
        tokenizer.save(tok_path)

    encoded_data = tokenizer.encode(raw_text)
    data_tensor = torch.tensor(encoded_data, dtype=torch.long)

    # 1. Initialize the Raw Model Architecture
    raw_model = QuantCB_Model(
        vocab_size=vocab_size, 
        d_model=256, 
        n_heads=8, 
        d_ff=512,        
        n_layers=4,
        latent_dim=128, 
        head_dim=64,
        num_experts=8,
        top_k=2
    )
    
    # 2. Wrap in the Execution Engine
    engine = QuantCB_Engine(raw_model).to(device)
    
    # 3. Optimization Strategy
    # We use a slightly higher LR with Cosine Decay for faster CPU convergence
    optimizer = optim.AdamW(engine.parameters(), lr=5e-4, weight_decay=0.01)
    
    # Cosine Annealing helps the experts settle in during the final steps
    num_iters = 501
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iters)
    
    print(f"--- Training QuantCB (DeepSeek-V3 Style) ---")
    print(f"Device: {device} | Iterations: {num_iters}")
    print("Features: MLA, Sparse MoE, and Multi-Token Prediction (MTP)")

    engine.train()
    for iter in range(num_iters): 
        # xb: [batch, 128], yb: [batch, 128] (shifted by 1)
        xb, yb = get_batch(data_tensor, 32, 128, device)
        
        # Engine handles the internal MTP slicing and combined loss
        logits, loss, _ = engine(xb, yb) 
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        # Clip gradients to prevent MoE router explosions
        torch.nn.utils.clip_grad_norm_(engine.parameters(), 1.0)
        
        optimizer.step()
        scheduler.step()

        if iter % 20 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Step {iter:4d} | Total Loss: {loss.item():.4f} | LR: {current_lr:.6f}")

    # 4. Save Weights
    # We save raw_model.state_dict() so it's compatible with inference/FP8 scripts
    save_path = os.path.join(output_dir, 'quantcb_mtp.pth')
    torch.save(raw_model.state_dict(), save_path)
    print(f"\nTraining complete. Weights saved to {save_path}")

if __name__ == "__main__":
    train()