import os
import urllib.request
import torch
import torch.optim as optim
from models.quantcb_model import QuantCB_Model
from tokenizer_basic import QuantCB_Tokenizer

def get_batch(data, batch_size, block_size, device):
    # We need block_size + 1 to provide the targets for n+2 prediction
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

def train():
    # Portable device selection
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(src_dir)
    output_dir = os.path.join(project_root, "modelOutput")
    os.makedirs(output_dir, exist_ok=True)

    data_path = os.path.join(output_dir, "input.txt")
    if not os.path.exists(data_path):
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        urllib.request.urlretrieve(url, data_path)
    
    with open(data_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()

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

    # Model definition with MTP enabled
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
    
    print(f"Training initialized on {device}.")
    print("Features: MLA, Sparse MoE, and Multi-Token Prediction (MTP)")

    model.train()
    for iter in range(501): 
        xb, yb = get_batch(data_tensor, 32, 128, device)
        
        # Forward pass returns the combined loss (Main + MTP)
        logits, loss, _ = model(xb, yb) 
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if iter % 20 == 0:
            # If you want to see the real progress, split them:
            with torch.no_grad():
                # This is a rough estimate for logging
                print(f"Step {iter} | Total: {loss.item():.4f} | Target Main Loss: ~3.0")

    save_path = os.path.join(output_dir, 'quantcb_mtp.pth')
    torch.save(model.state_dict(), save_path)
    print(f"Training complete. Weights saved to {save_path}")

if __name__ == "__main__":
    train()