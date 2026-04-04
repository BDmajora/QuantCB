import os
import torch
import torch.nn.functional as F
import time
from config import *
from lr_scheduler import get_lr
from data_engine import load_and_tag_all_data, get_batch
from models.quantcb_model import QuantCB_Model
from models.ouro_engine import Ouro_Engine 
from tokenizer import Tokenizer

# Define a path for the tokenizer cache inside modelOutput
TOKENIZER_PATH = os.path.join(OUTPUT_DIR, "quantcb_tokenizer.json")

def train():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load Data
    raw_text = load_and_tag_all_data()
    
    # 2. Tokenizer Lifecycle (Load or Train)
    tokenizer = Tokenizer(device="cpu") # Multi-threaded CPU is best for BPE
    
    if tokenizer.load(TOKENIZER_PATH):
        print("Using cached tokenizer.")
    else:
        print("No cached tokenizer found. Starting fresh BPE training...")
        # Sample 1M characters for representative vocab learning
        # This prevents the X13s from hanging on the full Wikipedia dump
        train_text_sample = raw_text[:1_000_000] 
        tokenizer.train(train_text_sample, target_vocab_size=VOCAB_SIZE)
        tokenizer.save(TOKENIZER_PATH)
    
    # 3. Encode Dataset (Vectorized via TokenEngine)
    print("Encoding full dataset...")
    train_data_list = tokenizer.encode(raw_text)
    train_data = torch.tensor(train_data_list, dtype=torch.long)
    print(f"Dataset Size: {len(train_data)} tokens")

    # 4. Identification Logic
    # Extract the ID for the hallucinate tag to supervise the Ouro Probe
    hallucinate_tokens = tokenizer.encode(TAGS["hallucinate"])
    hallucinate_id = hallucinate_tokens[0] if hallucinate_tokens else 0

    # 5. Model and Engine Setup
    raw_model = QuantCB_Model(
        vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_layers=N_LAYERS, 
        num_experts=NUM_EXPERTS, top_k=TOP_K
    )
    
    engine = Ouro_Engine(
        raw_model, 
        max_loops=MAX_LOOPS, 
        exit_threshold=EXIT_THRESHOLD
    ).to(DEVICE)
    
    optimizer = torch.optim.AdamW(
        engine.parameters(), 
        lr=MAX_LR, 
        weight_decay=WEIGHT_DECAY
    )

    start_iter = 0
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Resuming from checkpoint: {CHECKPOINT_PATH}")
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        engine.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_iter = ckpt['iteration'] + 1

    print(f"--- Phase 2 Active: {MAX_LOOPS} Loops | {EXIT_THRESHOLD} Entropy Exit ---")
    
    try:
        t0 = time.time()
        for iter in range(start_iter, ITERATIONS):
            # Dynamic Learning Rate
            lr = get_lr(iter, ITERATIONS)
            for pg in optimizer.param_groups: pg['lr'] = lr

            # A. Get Batch
            xb, yb = get_batch(train_data, BATCH_SIZE, BLOCK_SIZE, DEVICE)
            
            # B. Supervised Signal for Thinking Gate
            # We identify sequences containing the hallucinate tag
            is_corrupted = (xb == hallucinate_id).any(dim=1, keepdim=True)
            drift_targets = is_corrupted.expand(-1, BLOCK_SIZE).float().to(DEVICE)
            
            # C. Forward pass
            # Returns: final_logits, cross_entropy_loss, [list_of_probe_logits]
            logits, loss, probe_logits_list = engine(xb, yb)
            
            # D. Latent Probe Loss (Averaged over recursion loops)
            # This teaches the model to recognize "noise" internally
            probe_logits = torch.stack(probe_logits_list).mean(dim=0)
            bce_loss = F.binary_cross_entropy_with_logits(
                probe_logits.view(-1), 
                drift_targets.view(-1)
            )
            
            # Total Loss: Language Modeling + Thought Supervision
            total_loss = loss + (0.5 * bce_loss)
            
            # E. Backward Pass
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(engine.parameters(), GRAD_CLIP)
            optimizer.step()

            # F. Logging
            if iter % 10 == 0:
                t1 = time.time()
                dt = t1 - t0 
                # Tokens Per Second calculation
                tps = (BATCH_SIZE * BLOCK_SIZE * 10) / dt
                print(f"Step {iter:4d} | Loss: {total_loss.item():.4f} (Probe: {bce_loss.item():.4f}) | LR: {lr:.2e} | TPS: {tps:.0f}")
                t0 = t1

            # G. Generation Health Check
            if iter % 50 == 0 and iter > 0:
                engine.eval()
                print(f"\n--- Health Check Step {iter} ---")
                with torch.no_grad():
                    seed_str = f"{TAGS['truth']}{TAGS['stories']} "
                    context_ids = tokenizer.encode(seed_str)
                    context = torch.tensor([context_ids], dtype=torch.long, device=DEVICE)
                    generated = engine.generate(context, max_new_tokens=40)
                    print(f"Output: {tokenizer.decode(generated[0].tolist())}\n")
                engine.train()

            # H. Periodic Saving
            if iter % 50 == 0:
                torch.save({
                    'iteration': iter, 
                    'model_state_dict': engine.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, CHECKPOINT_PATH)

    except KeyboardInterrupt:
        print("\n--- Saving session state before exit ---")
        torch.save({
            'iteration': iter, 
            'model_state_dict': engine.state_dict(), 
            'optimizer_state_dict': optimizer.state_dict()
        }, CHECKPOINT_PATH)

if __name__ == "__main__":
    train()