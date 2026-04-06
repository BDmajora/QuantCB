import os
import torch
import numpy as np
from config import *
from lr_scheduler import get_lr
from models.quantcb_model import QuantCB_Model
from models.ouro_engine import Ouro_Engine 
from tokenizer import Tokenizer

# --- CORE PINNING & MEMORY OPTIMIZATION ---
# Since DEVICE is gone, we optimize CPU threading globally. 
# This prevents the Python/MKL backend from fighting for logical cores 
# while the Vulkan kernels are being dispatched.
physical_cores = max(1, os.cpu_count() // 2)
torch.set_num_threads(physical_cores)

# Boost precision for CPU-side weight prep and optimizer math
torch.set_float32_matmul_precision('high')

TOKENIZER_PATH = os.path.join(OUTPUT_DIR, "quantcb_tokenizer.json")

def setup_tokenizer(raw_text):
    """Loads cached tokenizer or trains a new one."""
    tokenizer = Tokenizer(device="cpu")
    if tokenizer.load(TOKENIZER_PATH):
        print(f"Using cached tokenizer from {TOKENIZER_PATH}")
    else:
        print("No cached tokenizer found. Starting fresh BPE training...")
        train_text_sample = raw_text[:1_000_000] 
        tokenizer.train(train_text_sample, target_vocab_size=VOCAB_SIZE)
        tokenizer.save(TOKENIZER_PATH)
    return tokenizer

def encode_dataset(tokenizer, raw_text):
    """
    Encodes text and pins the memory. 
    Pinned memory is essential for high-speed DMA transfers to your RX 6800.
    """
    print("Encoding full dataset...")
    train_data_list = tokenizer.encode(raw_text)
    
    # We move to a tensor immediately and use pin_memory=True 
    # to bypass the CPU-to-Vulkan staging bottleneck.
    train_data = torch.tensor(train_data_list, dtype=torch.long, pin_memory=True)
    
    print(f"Dataset Size: {len(train_data)} tokens (Pinned to CPU Memory)")
    return train_data

def setup_model(device_str):
    """
    Initializes the model and engine.
    device_str: Pass your 'vulkan' or 'cuda' or 'cpu' target here.
    """
    raw_model = QuantCB_Model(
        vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_layers=N_LAYERS, 
        num_experts=NUM_EXPERTS, top_k=TOP_K
    )
    
    engine = Ouro_Engine(
        raw_model, 
        max_loops=MAX_LOOPS, 
        exit_threshold=EXIT_THRESHOLD
    ).to(device_str)

    print(f"Engine initialized in Eager Mode (CPU Threads: {torch.get_num_threads()})")
    
    optimizer = torch.optim.AdamW(
        engine.parameters(), 
        lr=MAX_LR, 
        weight_decay=WEIGHT_DECAY
    )
    return engine, optimizer

def load_checkpoint(engine, optimizer, device_str):
    """Loads model state and returns the next iteration count."""
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Resuming from checkpoint: {CHECKPOINT_PATH}")
        # weights_only=True is faster and safer for state_dict loads
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device_str, weights_only=True)
        engine.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        return ckpt['iteration'] + 1
    return 0

def update_lr(optimizer, step, start_iter):
    """Updates the optimizer's LR based on the Cosine/Warmup scheduler."""
    current_lr = get_lr(step, start_iter)
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr
    return current_lr