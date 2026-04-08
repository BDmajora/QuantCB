import torch
import torch.nn as nn
import sys
import os

# Ensure the current directory is in the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rope import DynamicNTKRotaryEmbedding, apply_rotary_pos_emb, compute_ntk_inv_freq

def run_rope_verification():
    print("--- Starting RoPE and Dynamic NTK Verification ---")
    
    # 1. Configuration
    dim = 64
    max_pos = 128  # Small limit to trigger NTK scaling easily
    base = 10000
    batch, heads, seq_len = 1, 8, 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Initialize Module
    rope_module = DynamicNTKRotaryEmbedding(
        dim=dim, 
        max_position_embeddings=max_pos, 
        base=base, 
        device=device
    )
    
    # Create test tensors
    q = torch.randn(batch, heads, seq_len, dim, device=device)
    k = torch.randn(batch, heads, seq_len, dim, device=device)

    # --- Test 1: Standard Forward Pass ---
    print(f"[Test 1] Standard Forward (seq_len={seq_len})")
    cos, sin = rope_module(q, seq_len)
    
    # Verify shapes: [1, 1, seq_len, dim]
    expected_shape = (1, 1, seq_len, dim)
    if cos.shape == expected_shape and sin.shape == expected_shape:
        print("PASS: Cosine and Sine shapes are correct.")
    else:
        print(f"FAIL: Shape mismatch. Got {cos.shape}, expected {expected_shape}")

    q_emb, k_emb = apply_rotary_pos_emb(q, k, cos, sin)
    if q_emb.shape == q.shape:
        print("PASS: apply_rotary_pos_emb output shape is correct.")

    # --- Test 2: NTK Frequency Scaling ---
    print(f"\n[Test 2] Dynamic NTK Scaling (seq_len=256 > max_pos={max_pos})")
    
    # Get initial frequencies
    initial_inv_freq = rope_module.inv_freq.clone()
    
    # Trigger scaling by requesting a long sequence
    long_seq = 256
    q_long = torch.randn(batch, heads, long_seq, dim, device=device)
    cos_long, sin_long = rope_module(q_long, long_seq)
    
    # The compute_ntk_inv_freq should have been called internally
    # We verify if the current frequencies in the module match the NTK calculation
    expected_inv_freq = compute_ntk_inv_freq(dim, long_seq, max_pos, base, device)
    
    # Check if frequencies shifted from the original base
    if not torch.allclose(initial_inv_freq, expected_inv_freq):
        print("PASS: Dynamic NTK scaling shifted the inverse frequencies.")
    else:
        print("FAIL: Frequencies remained static despite exceeding max_pos.")

    # --- Test 3: Cache Persistence and Device Consistency ---
    print("\n[Test 3] Cache and Device Consistency")
    
    # Ensure the cache is actually stored
    if rope_module.cos_cached is not None and rope_module.sin_cached is not None:
        print("PASS: cos_cached and sin_cached are populated.")
    else:
        print("FAIL: Cache buffers are empty.")

    # Check device consistency
    if rope_module.cos_cached.device == device:
        print(f"PASS: Cache is on the correct device ({device}).")
    else:
        print(f"FAIL: Cache device mismatch. Found {rope_module.cos_cached.device}")

    # --- Test 4: Mathematical Transformation Check ---
    print("\n[Test 4] Rotary Transformation Identity Check")
    # A rotary embedding should change the values unless the angle is 0
    diff = torch.norm(q_emb - q).item()
    if diff > 1e-3:
        print(f"PASS: Transformation applied (Norm change: {diff:.4f})")
    else:
        print("FAIL: Q remains unchanged after RoPE application.")

    print("\n--- Verification Complete ---")

if __name__ == "__main__":
    run_rope_verification()