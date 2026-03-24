import torch
import time
import os
from models.quantcb_model import QuantCB_Model

def benchmark():
    vocab_size, d_model, n_layers = 50257, 256, 4
    dummy_input = torch.randint(0, vocab_size, (1, 128)) # Full context window
    
    # 1. Benchmark FP32
    model_fp32 = QuantCB_Model(vocab_size, d_model, n_layers=n_layers)
    model_fp32.load_state_dict(torch.load('quantcb_base.pth'))
    model_fp32.eval()
    
    start = time.time()
    for _ in range(50):
        with torch.no_grad(): _ = model_fp32(dummy_input)
    fp32_time = (time.time() - start) / 50
    
    # 2. Benchmark INT8 (Simulated via your inference script logic)
    # Note: Real speedup requires specialized INT8 kernels, but we can measure 
    # the memory footprint and the dequantization overhead.
    fp32_size = os.path.getsize('quantcb_base.pth') / 1e6
    int8_size = os.path.getsize('quantcb_int8.pth') / 1e6
    
    print(f"\n--- QuantCB Benchmark Results ---")
    print(f"FP32 Model Size: {fp32_size:.2f} MB")
    print(f"INT8 Model Size: {int8_size:.2f} MB ({(fp32_size/int8_size):.2f}x Compression)")
    print(f"Average Latency (FP32): {fp32_time*1000:.2f} ms")
    print(f"--- Benchmark Complete ---")

if __name__ == "__main__":
    benchmark()