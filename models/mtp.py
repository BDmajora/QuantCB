import torch
import torch.nn.functional as F

def mtp_forward_stateless(
    h_base: torch.Tensor,
    targets: torch.Tensor,
    embedding_weight: torch.Tensor,
    head_weight: torch.Tensor,
    proj_h_weight: torch.Tensor,
    proj_emb_weight: torch.Tensor,
    ln_fusion_weight: torch.Tensor,
    ln_fusion_bias: torch.Tensor,
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
    n_heads: int
):
    B, T, d_model = h_base.shape
    vocab_size = head_weight.shape[0]

    # 1. Get embeddings for the 'hint' tokens (t+1)
    x_embed = F.embedding(targets, embedding_weight)
    
    # 2. DeepSeek-V3 style additive fusion
    fused_h = F.linear(h_base, proj_h_weight, bias=None)
    fused_emb = F.linear(x_embed, proj_emb_weight, bias=None)
    fused = (fused_h + fused_emb) * 0.5
    
    x = F.layer_norm(fused, (d_model,), weight=ln_fusion_weight, bias=ln_fusion_bias)

    # 3. Transformer Block
    # --- 3a. Norm 1 ---
    x_norm1 = F.layer_norm(x, (d_model,), weight=norm1_weight, bias=norm1_bias)
    
    # --- 3b. Self-Attention ---
    qkv = F.linear(x_norm1, attn_qkv_weight, attn_qkv_bias)
    q, k, v = qkv.chunk(3, dim=-1)
    
    head_dim = d_model // n_heads
    q = q.view(B, T, n_heads, head_dim).transpose(1, 2)
    k = k.view(B, T, n_heads, head_dim).transpose(1, 2)
    v = v.view(B, T, n_heads, head_dim).transpose(1, 2)
    
    attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, d_model)
    attn_output = F.linear(attn_output, attn_out_weight, attn_out_bias)
    
    x = x + attn_output # Residual 1
    
    # --- 3c. Norm 2 ---
    x_norm2 = F.layer_norm(x, (d_model,), weight=norm2_weight, bias=norm2_bias)
    
    # --- 3d. MLP ---
    mlp_out = F.linear(x_norm2, mlp_fc1_weight, mlp_fc1_bias)
    mlp_out = F.gelu(mlp_out)
    mlp_out = F.linear(mlp_out, mlp_fc2_weight, mlp_fc2_bias)
    
    x_mtp = x + mlp_out # Residual 2
    
    # 4. Predict t+2 using the shared head
    logits = F.linear(x_mtp, head_weight, bias=None) 
    
    # --- NEW: Calculate Loss ---
    # We flatten logits and targets to use standard CrossEntropy
    # Note: In a real MTP setup, you might need to shift targets 
    # depending on if 'targets' represents t+1 or t+2.
    loss = F.cross_entropy(
        logits.view(-1, vocab_size), 
        targets.view(-1)
    )
    
    return logits, loss # Return the loss instead of hidden states