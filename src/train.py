import os
from config import *
from data_engine import load_and_tag_all_data
from setup_utils import setup_tokenizer, encode_dataset, setup_iree_runtime, load_checkpoint
from trainer import QuantCBTrainer

def train():
    """
    IREE AOT Execution Entry Point.
    PyTorch is only used here as a data-loader backend.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load and Tag Data (CPU Host)
    raw_text = load_and_tag_all_data()
    
    # 2. Tokenize into NumPy Buffers
    tokenizer = setup_tokenizer(raw_text)
    train_data = encode_dataset(tokenizer, raw_text)
    
    # Clear raw text to free up host RAM for the Vulkan driver
    del raw_text 
    
    # 3. Initialize IREE Vulkan Runtime
    engine, config = setup_iree_runtime()
    
    # 4. Resume State
    start_iter = load_checkpoint()
    
    # 5. Execute Trainer Loop
    trainer = QuantCBTrainer(
        engine=engine,
        config=config,
        train_data=train_data,
        tokenizer=tokenizer,
        start_iter=start_iter,
    )
    
    print(f"--- Launching IREE Vulkan Pipeline from Step {start_iter} ---")
    trainer.run()

if __name__ == "__main__":
    train()