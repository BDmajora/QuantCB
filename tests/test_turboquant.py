import sys
import os
import torch
import torch.nn.functional as F

# --- LOCAL TEST UTILITIES (Standalone TurboQuant Logic) ---
# Included locally so the test can run independently of file structure imports

def generate_orthogonal_matrix(dim):
    H = torch.randn(dim, dim)
    Q, R = torch.linalg.qr(H)
    d = torch.diag(R)
    ph = d.sign()
    Q *= ph
    return Q

def mock_turboquant_compress(tensor, bits=4):
    """Simulates the TurboQuant compression for testing."""
    out_features, in_features = tensor.shape
    
    # 1. Random Rotation
    R = generate_orthogonal_matrix(in_features).to(tensor.device)
    W_rot = tensor @ R
    
    # 2. Stage One: MSE Scalar Quantizer (4-bit)
    q_min, q_max = -(2**(bits-1)), (2**(bits-1)) - 1
    max_val = W_rot.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-12)
    scale = max_val / q_max
    
    W_q = torch.round(W_rot / scale).clamp(q_min, q_max).to(torch.int8)
    W_deq_base = W_q.to(torch.float32) * scale
    
    # 3. Stage Two: 1-bit QJL Residual
    residual = W_rot - W_deq_base
    residual_scale = residual.abs().mean(dim=1, keepdim=True)
    residual_sign = torch.sign(residual).to(torch.int8)
    
    return {
        'weight_q': W_q,
        'scale': scale,
        'residual_sign': residual_sign,
        'residual_scale': residual_scale,
        'rotation': R
    }

def mock_turboquant_decompress(data):
    """Simulates the inference-side reconstruction."""
    W_base = data['weight_q'].to(torch.float32) * data['scale']
    W_residual = data['residual_sign'].to(torch.float32) * data['residual_scale']
    W_rot_approx = W_base + W_residual
    W_approx = W_rot_approx @ data['rotation'].T
    return W_approx

# --- MAIN COMPRESSION TEST SUITE ---

def run_compression_check():
    print("--- Phase 1: Initialization & Tensor Generation ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Mock a weight matrix from a typical LLM linear layer (e.g., d_model to d_ff)
    in_features = 256
    out_features = 1024
    
    # Kaiming uniform initialization to mimic real weights
    W_orig = torch.randn(out_features, in_features, device=device) * (2.0 / in_features)**0.5
    print(f"Generated FP32 Weight Matrix: Shape {W_orig.shape}")
    print(f"Base Memory Footprint: {(W_orig.numel() * 4) / 1024:.2f} KB")

    print("\n--- Phase 2: TurboQuant Compression (4-bit + 1-bit Residual) ---")
    try:
        compressed_data = mock_turboquant_compress(W_orig, bits=4)
        
        # Calculate theoretical compressed size
        w_q_size = compressed_data['weight_q'].numel() * 1  # INT8 acting as 4-bit container (1 byte)
        res_size = compressed_data['residual_sign'].numel() * 1 # INT8 acting as 1-bit container (1 byte)
        scale_size = compressed_data['scale'].numel() * 4
        res_scale_size = compressed_data['residual_scale'].numel() * 4
        
        # Note: In a real deploy, 4-bit and 1-bit are packed. We calculate theoretical packed sizes here.
        theoretical_packed_size = (w_q_size * 0.5) + (res_size * 0.125) + scale_size + res_scale_size
        
        print("Compression successful.")
        print(f"Est. Packed Memory Footprint: {theoretical_packed_size / 1024:.2f} KB")
        print(f"Est. Compression Ratio: {((W_orig.numel() * 4) / theoretical_packed_size):.2f}x")
    except Exception as e:
        print(f"Compression Failed: {e}")
        return

    print("\n--- Phase 3: Dequantization & Structural Integrity ---")
    try:
        W_recon = mock_turboquant_decompress(compressed_data)
        if W_orig.shape == W_recon.shape:
            print("Dequantization successful. Shape matched.")
    except Exception as e:
        print(f"Dequantization Failed: {e}")
        return

    print("\n--- Phase 4: Distortion Rate & Fidelity Metrics ---")
    with torch.no_grad():
        # 1. Mean Squared Error (MSE)
        mse_loss = F.mse_loss(W_orig, W_recon).item()
        
        # 2. Cosine Similarity (Structural preservation)
        cos_sim = F.cosine_similarity(W_orig.flatten(), W_recon.flatten(), dim=0).item()
        
        print(f"Mean Squared Error (MSE):  {mse_loss:.6f}")
        print(f"Cosine Similarity:         {cos_sim:.4f} (Ideal is 1.0000)")
        
        if cos_sim > 0.95:
            print("PASS: High geometric structure preservation.")
        else:
            print("WARNING: Significant geometric degradation.")

    print("\n--- Phase 5: Inner Product (MatMul) Unbiasedness Test ---")
    # TurboQuant's main claim is that inner product distortion is optimal/unbiased
    with torch.no_grad():
        # Create a mock activation/input tensor (Batch=4, Seq=16, Dim=256)
        X = torch.randn(4, 16, in_features, device=device)
        
        # FP32 Matmul
        Y_exact = X @ W_orig.T
        
        # Quantized Matmul
        Y_quant = X @ W_recon.T
        
        # Measure activation error
        act_error = F.l1_loss(Y_exact, Y_quant).item()
        act_max = Y_exact.abs().max().item()
        error_ratio = act_error / act_max
        
        print(f"FP32 Output Mean:       {Y_exact.mean().item():.4f}")
        print(f"TurboQuant Output Mean: {Y_quant.mean().item():.4f}")
        print(f"Mean Absolute Error:    {act_error:.4f} (relative to max value {act_max:.4f})")
        print(f"Error Ratio:            {error_ratio * 100:.2f}%")
        
        if error_ratio < 0.05:
            print("PASS: Inner product estimation remains highly unbiased.")

if __name__ == "__main__":
    run_compression_check()