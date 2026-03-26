import os
import torch
import torch.optim as optim
import math
from models.quantcb_model import QuantCB_Model

def train():
    # 1. Hardware Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 2. Hyperparameters
    vocab_size = 50257 
    d_model = 256
    n_layers = 4
    n_heads = 8
    d_ff = 1024
    latent_dim = 128
    head_dim = 64
    
    batch_size = 32
    block_size = 128 
    learning_rate = 3e-4

    # 3. Model Initialization
    model = QuantCB_Model(
        vocab_size=vocab_size, 
        d_model=d_model, 
        n_heads=n_heads, 
        d_ff=d_ff, 
        n_layers=n_layers,
        latent_dim=latent_dim,
        head_dim=head_dim
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)

    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model initialized on {device}. Parameters: {params_m:.2f}M")

    # 4. Structural Verification Loop
    model.train()
    for iter in range(101):
        xb = torch.randint(0, vocab_size, (batch_size, block_size)).to(device)
        yb = torch.randint(0, vocab_size, (batch_size, block_size)).to(device)

        logits, loss = model(xb, yb)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if iter % 10 == 0:
            print(f"Step {iter}: Loss {loss.item():.4f}")

    # 5. Checkpoint Persistence to /modelOutput
    # Find the project root (one level up from /src)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, "modelOutput")
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    save_path = os.path.join(output_dir, 'quantcb_base.pth')
    torch.save(model.state_dict(), save_path)
    
    print(f"\nTraining smoke test complete. Checkpoint saved to {save_path}")

if __name__ == "__main__":
    train()