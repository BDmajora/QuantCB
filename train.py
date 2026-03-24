import torch
import torch.optim as optim
import math
from models.quantcb_model import QuantCB_Model

# from tokenizer_basic import BPETokenizer 

def train():
    # 1. Hardware Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 2. Hyperparameters
    vocab_size = 50257 
    d_model = 256
    n_layers = 4
    n_heads = 8
    d_ff = 1024
    batch_size = 32
    block_size = 128 
    learning_rate = 3e-4

    # 3. Model Initialization
    model = QuantCB_Model(
        vocab_size=vocab_size, 
        d_model=d_model, 
        n_heads=n_heads, 
        d_ff=d_ff, 
        n_layers=n_layers
    ).to(device)
    
    # AdamW with Weight Decay for Transformer stability
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)

    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model initialized on {device}. Parameters: {params_m:.2f}M")

    # 4. Structural Verification Loop (Smoke Test)
    model.train()
    for iter in range(101):
        # Synthetic data generation for graph verification
        xb = torch.randint(0, vocab_size, (batch_size, block_size)).to(device)
        yb = torch.randint(0, vocab_size, (batch_size, block_size)).to(device)

        # Forward Pass
        logits, loss = model(xb, yb)
        
        # Backward Pass & Optimization
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if iter % 10 == 0:
            print(f"Step {iter}: Loss {loss.item():.4f}")

    # 5. Checkpoint Persistence
    torch.save(model.state_dict(), 'quantcb_base.pth')
    print("\n✅ Training smoke test complete. Checkpoint saved to quantcb_base.pth")

if __name__ == "__main__":
    train()