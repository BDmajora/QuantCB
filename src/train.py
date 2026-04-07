import os
import iree.runtime as ireert
from pathlib import Path
from config import *
from data_engine import load_and_tag_all_data
from setup_utils import setup_tokenizer, encode_dataset
from trainer_vulkan import IREEVulkanTrainer

def train():
    """
    Pure Vulkan Training Entry Point.
    Decoupled from PyTorch runtime; executes SPIR-V kernels directly.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Prepare Data (CPU)
    print("--- 2026 QuantCB: Data Preparation ---")
    raw_text = load_and_tag_all_data()
    tokenizer = setup_tokenizer(raw_text)
    train_data = encode_dataset(tokenizer, raw_text)
    del raw_text # Minimize CPU RAM usage
    
    # 2. Initialize Vulkan Device
    # This reaches out to the RX 6800 via the Vulkan SDK
    config = ireert.Config("vulkan")
    
    # 3. Locate the Training Binary
    # This file must contain the baked training logic
    vmfb_path = Path(OUTPUT_DIR) / "quantcb_vulkan_train.vmfb"
    
    if not vmfb_path.exists():
        print(f"ERROR: {vmfb_path} not found. Ensure you compiled the training module.")
        return

    # 4. Launch the Native Trainer
    trainer = IREEVulkanTrainer(
        config=config,
        vmfb_path=vmfb_path,
        train_data=train_data,
        tokenizer=tokenizer,
        start_iter=0
    )
    
    print(f"--- Starting Native SPIR-V Training Pipeline ---")
    trainer.run()

if __name__ == "__main__":
    train()