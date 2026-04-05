import os
import torch
from config import *
from data_engine import load_and_tag_all_data
from models.quantcb_model import QuantCB_Model
from models.ouro_engine import Ouro_Engine 
from tokenizer import Tokenizer

# --- CORE PINNING OPTIMIZATION ---
# Restricting PyTorch to physical cores prevents "thread thrashing."
# On many-core CPUs, using half the logical count (physical count) 
# typically maximizes the compute-to-overhead ratio for MoE models.
if DEVICE == 'cpu':
    physical_cores = max(1, os.cpu_count() // 2)
    torch.set_num_threads(physical_cores)

# Optimization: Modern CPUs (Intel/AMD) benefit from higher precision math 
# settings even when using bfloat16 autocast.
torch.set_float32_matmul_precision('high')

TOKENIZER_PATH = os.path.join(OUTPUT_DIR, "quantcb_tokenizer.json")

def setup_tokenizer(raw_text):
    """Loads cached tokenizer or trains a new one."""
    tokenizer = Tokenizer(device="cpu")
    if tokenizer.load(TOKENIZER_PATH):
        print("Using cached tokenizer.")
    else:
        print("No cached tokenizer found. Starting fresh BPE training...")
        train_text_sample = raw_text[:1_000_000] 
        tokenizer.train(train_text_sample, target_vocab_size=VOCAB_SIZE)
        tokenizer.save(TOKENIZER_PATH)
    return tokenizer

def encode_dataset(tokenizer, raw_text):
    """Encodes text to tensors and frees string memory."""
    print("Encoding full dataset...")
    train_data_list = tokenizer.encode(raw_text)
    train_data = torch.tensor(train_data_list, dtype=torch.long)
    print(f"Dataset Size: {len(train_data)} tokens")
    return train_data

def setup_model(device):
    """Initializes the raw model and engine in Eager Mode for CPU throughput."""
    raw_model = QuantCB_Model(
        vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_layers=N_LAYERS, 
        num_experts=NUM_EXPERTS, top_k=TOP_K
    )
    
    engine = Ouro_Engine(
        raw_model, 
        max_loops=MAX_LOOPS, 
        exit_threshold=EXIT_THRESHOLD
    ).to(device)

    # --- NO COMPILATION ---
    # We are using Eager Mode to avoid the deadlock/hang. 
    # Performance is regained via vectorized batching and core-pinning.
    print(f"Engine initialized in Eager Mode (Threads: {torch.get_num_threads()})")
    
    optimizer = torch.optim.AdamW(
        engine.parameters(), 
        lr=MAX_LR, 
        weight_decay=WEIGHT_DECAY
    )
    return engine, optimizer

def load_checkpoint(engine, optimizer, device):
    """Loads model state if a checkpoint exists."""
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Resuming from checkpoint: {CHECKPOINT_PATH}")
        # Use map_location to ensure weights load directly to the intended device
        # Added weights_only=True for security/best practice in newer PyTorch
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
        engine.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        return ckpt['iteration'] + 1
    return 0