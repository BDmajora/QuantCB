import os
import torch
import numpy as np
# 2026 Standard: IREE Runtime handles the execution of the compiled binary
import iree.runtime as ireert 
from config import *
from lr_scheduler import get_lr
from tokenizer import Tokenizer

# CPU is now ONLY used for data orchestration and Tokenization.
physical_cores = max(1, os.cpu_count() // 2)
torch.set_num_threads(physical_cores)

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
    Encodes text into standard NumPy arrays.
    IREE's Vulkan runtime prefers NumPy buffers for zero-copy memory mapping
    rather than PyTorch pinned tensors.
    """
    print("Encoding full dataset...")
    train_data_list = tokenizer.encode(raw_text)
    train_data = np.array(train_data_list, dtype=np.int32)
    print(f"Dataset Size: {len(train_data)} tokens (Host RAM)")
    return train_data

def setup_iree_runtime():
    """
    Loads the AOT-compiled Vulkan module.
    No PyTorch eager models are initialized here.
    """
    vmfb_path = os.path.join(OUTPUT_DIR, "quantcb_vulkan.vmfb")
    if not os.path.exists(vmfb_path):
        raise FileNotFoundError(f"Missing compiled binary: {vmfb_path}. Run compile_for_vulkan.py first.")

    print(f"Loading IREE VM FlatBuffer from {vmfb_path}...")
    
    # Initialize the Vulkan HAL (Hardware Abstraction Layer)
    config = ireert.Config("vulkan")
    
    # Memory-map the binary directly via the VmModule class for instant load times
    vm_module = ireert.VmModule.mmap(config.vm_instance, vmfb_path)
    
    # Bind the module to the Vulkan context
    ctx = ireert.SystemContext(config=config)
    ctx.add_vm_module(vm_module)
    
    # 'module' is the compiled PyTorch graph
    engine = ctx.modules.module 
    
    print("--- IREE Vulkan Runtime Active ---")
    return engine, config

def load_checkpoint(start_iter=0):
    """
    In the AOT paradigm, state dicts are managed directly by the IREE module 
    or loaded via side-channels. For simplicity, we track iteration count.
    """
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Resuming metadata from: {CHECKPOINT_PATH}")
        ckpt = torch.load(CHECKPOINT_PATH, weights_only=True)
        return ckpt.get('iteration', 0) + 1
    return start_iter