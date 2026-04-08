import torch
import torch.nn.functional as F
import traceback

def mtp_forward_stateless(
    h_base: torch.Tensor,
    targets: torch.Tensor,
    # --- Shared Weights ---
    embedding_weight: torch.Tensor,
    head_weight: torch.Tensor,
    # --- Fusion Weights ---
    proj_h_weight: torch.Tensor,
    proj_emb_weight: torch.Tensor,
    ln_fusion_weight: torch.Tensor,
    ln_fusion_bias: torch.Tensor,
    # --- Transformer Block Weights (Norm-First) ---
    norm1_weight: torch.Tensor,
    norm1_bias: torch.Tensor,
    attn_qkv_weight: torch.Tensor,
    attn_qkv_bias: torch.Tensor,
    attn_out_weight: torch.Tensor,
    attn_out_bias: torch.Tensor,
    norm2_weight: torch.Tensor,
    norm2_bias: torch.Tensor,
    mlp_fc1_weight: torch.Tensor,
    mlp_fc1_bias: torch.Tensor,
    mlp_fc2_weight: torch.Tensor,
    mlp_fc2_bias: torch.Tensor,
    # --- Architecture Config ---
    n_heads: int
):
    """
    Pure functional MTP forward pass. 
    All memory buffers (weights/biases) are passed in explicitly.
    """
    B, T, d_model = h_base.shape

    # 1. Get embeddings for the 'hint' tokens (t+1)
    x_embed = F.embedding(targets, embedding_weight)
    
    # 2. Mix: DeepSeek-V3 style additive fusion
    fused_h = F.linear(h_base, proj_h_weight, bias=None)
    fused_emb = F.linear(x_embed, proj_emb_weight, bias=None)
    fused = (fused_h + fused_emb) * 0.5
    
    x = F.layer_norm(fused, (d_model,), weight=ln_fusion_weight, bias=ln_fusion_bias)

    # 3. Transformer Layer (The Mixer)
    # --- 3a. Norm 1 ---
    x_norm1 = F.layer_norm(x, (d_model,), weight=norm1_weight, bias=norm1_bias)
    
    # --- 3b. Self-Attention ---
    qkv = F.linear(x_norm1, attn_qkv_weight, attn_qkv_bias)
    q, k, v = qkv.chunk(3, dim=-1)
    
    head_dim = d_model // n_heads
    q = q.view(B, T, n_heads, head_dim).transpose(1, 2)
    k = k.view(B, T, n_heads, head_dim).transpose(1, 2)
    v = v.view(B, T, n_heads, head_dim).transpose(1, 2)
    
    # SDPA handles causal masking automatically
    attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    
    attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, d_model)
    attn_output = F.linear(attn_output, attn_out_weight, attn_out_bias)
    
    x = x + attn_output # Residual 1
    
    # --- 3c. Norm 2 ---
    x_norm2 = F.layer_norm(x, (d_model,), weight=norm2_weight, bias=norm2_bias)
    
    # --- 3d. Feed-Forward Network (MLP) ---
    mlp_out = F.linear(x_norm2, mlp_fc1_weight, mlp_fc1_bias)
    mlp_out = F.gelu(mlp_out)
    mlp_out = F.linear(mlp_out, mlp_fc2_weight, mlp_fc2_bias)
    
    x_mtp = x + mlp_out # Residual 2
    
    # 4. Predict t+2 using the shared head
    logits = F.linear(x_mtp, head_weight, bias=None) 
    
    return logits, x_mtp


def verify_mtp_stateless():
    print("--- Starting MTP Functional Verification (Stateless) ---")
    
    # 1. Hyperparameters
    d_model, n_heads, d_ff = 512, 8, 2048
    vocab_size = 1000
    batch, seq = 2, 16

    # 2. Setup Mock Inputs
    # h_base: hidden states from main trunk
    h_base = torch.randn(batch, seq, d_model)
    # targets: tokens at t+1
    targets = torch.randint(0, vocab_size, (batch, seq))
    # labels: the actual tokens at t+2 we want to predict
    labels = torch.randint(0, vocab_size, (batch, seq))

    # 3. Initialize Mock Weights
    weights = {
        "embedding_weight": torch.randn(vocab_size, d_model) * 0.02,
        "head_weight": torch.randn(vocab_size, d_model) * 0.02,
        "proj_h_weight": torch.randn(d_model, d_model) * 0.02,
        "proj_emb_weight": torch.randn(d_model, d_model) * 0.02,
        "ln_fusion_weight": torch.ones(d_model),
        "ln_fusion_bias": torch.zeros(d_model),
        "norm1_weight": torch.ones(d_model),
        "norm1_bias": torch.zeros(d_model),
        "attn_qkv_weight": torch.randn(3 * d_model, d_model) * 0.02,
        "attn_qkv_bias": torch.zeros(3 * d_model),
        "attn_out_weight": torch.randn(d_model, d_model) * 0.02,
        "attn_out_bias": torch.zeros(d_model),
        "norm2_weight": torch.ones(d_model),
        "norm2_bias": torch.zeros(d_model),
        "mlp_fc1_weight": torch.randn(d_ff, d_model) * 0.02,
        "mlp_fc1_bias": torch.zeros(d_ff),
        "mlp_fc2_weight": torch.randn(d_model, d_ff) * 0.02,
        "mlp_fc2_bias": torch.zeros(d_model),
    }

    # 4. Run Functional Forward Pass
    try:
        with torch.no_grad():
            logits, _ = mtp_forward_stateless(
                h_base, 
                targets, 
                **weights, 
                n_heads=n_heads
            )

            # Calculate Loss for Verification
            loss = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1))

        # 5. Verification Checks
        print(f"[Test 1] Shape Verification")
        assert logits.shape == (batch, seq, vocab_size), f"Logits shape mismatch: {logits.shape}"
        print(f"PASS: Logits shape is {logits.shape}")

        print(f"\n[Test 2] Numerical Sanity")
        assert not torch.isnan(logits).any(), "NaN detected in logits!"
        assert not torch.isinf(logits).any(), "Inf detected in logits!"
        
        mean_loss = loss.item()
        assert mean_loss > 0, f"Loss should be positive, got {mean_loss}"
        print(f"PASS: Loss value is {mean_loss:.4f} (No NaNs/Infs)")

        print("\n--- ALL MTP TESTS PASSED ---")

    except Exception as e:
        print(f"\nFAILURE: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    verify_mtp_stateless()