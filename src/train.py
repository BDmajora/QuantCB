import os
import torch
from config import *
from data_engine import load_and_tag_all_data
from setup_utils import setup_tokenizer, encode_dataset, setup_model, load_checkpoint
from trainer import QuantCBTrainer

def train():
    """
    Modular training entry point. 
    Can be imported and executed by a separate Main class.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load and Tag Data (CPU)
    raw_text = load_and_tag_all_data()
    
    # 2. Tokenize to Pinned CPU Memory
    # This is the "Staging Area" for the RX 6800 Vulkan DMA
    tokenizer = setup_tokenizer(raw_text)
    train_data = encode_dataset(tokenizer, raw_text)
    
    # Clear raw text immediately to maximize available RAM for the Vulkan driver
    del raw_text 
    
    # 3. Model & Optimizer Setup
    # Using 'vulkan' target to bypass the standard Torch CPU overhead
    vulkan_device = "cpu" 
    engine, optimizer = setup_model(vulkan_device)
    
    # 4. Resume State
    start_iter = load_checkpoint(engine, optimizer, vulkan_device)
    
    # 5. Execute Trainer Loop
    # The trainer now handles the 'Vulkan Timeline' (Async dispatch)
    trainer = QuantCBTrainer(
        engine=engine,
        optimizer=optimizer,
        train_data=train_data,
        tokenizer=tokenizer,
        start_iter=start_iter,
        device=vulkan_device
    )
    
    print(f"--- Launching Vulkan Training Pipeline from Step {start_iter} ---")
    trainer.run()

if __name__ == "__main__":
    # If this file is run directly, start training
    train()