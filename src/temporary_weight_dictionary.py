import torch

def get_mock_weights(
    vocab_size, d_model, n_layers, n_heads, 
    head_dim, latent_dim, num_experts
):
    d_ff_block = 512 
    d_ff_mtp = d_model * 4 
    
    # Use .clone() to ensure these aren't the same object in memory
    shared_weight = torch.randn(vocab_size, d_model) * 0.02
    
    weights = {
        "token_embedding.weight": shared_weight.clone(),
        "lm_head.weight": shared_weight.clone(), 
        "ln_f.weight": torch.ones(d_model),
        "latent_probe.weight": torch.randn(1, d_model) * 0.02
    }

    for i in range(n_layers):
        weights[f"blocks.{i}.ln1.weight"] = torch.ones(d_model)
        weights[f"blocks.{i}.ln2.weight"] = torch.ones(d_model)
        
        weights[f"blocks.{i}.attn.wq_weight"] = torch.randn(n_heads * head_dim, d_model) * 0.02
        weights[f"blocks.{i}.attn.wdkv_weight"] = torch.randn(latent_dim, d_model) * 0.02
        weights[f"blocks.{i}.attn.ln_kv_weight"] = torch.ones(latent_dim)
        weights[f"blocks.{i}.attn.ln_kv_bias"] = torch.zeros(latent_dim)
        weights[f"blocks.{i}.attn.wuk_weight"] = torch.randn(n_heads * head_dim, latent_dim) * 0.02
        weights[f"blocks.{i}.attn.wuv_weight"] = torch.randn(n_heads * head_dim, latent_dim) * 0.02
        weights[f"blocks.{i}.attn.wo_weight"] = torch.randn(d_model, n_heads * head_dim) * 0.02
        
        # Ensure RoPE is detached and explicitly Float32
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        weights[f"blocks.{i}.attn.rope_inv_freq"] = inv_freq.detach().clone()

        weights[f"blocks.{i}.moe.router_weight"] = torch.randn(num_experts, d_model) * 0.02
        weights[f"blocks.{i}.moe.w1_weight"] = torch.randn(num_experts, d_ff_block, d_model) * 0.02 
        weights[f"blocks.{i}.moe.w3_weight"] = torch.randn(num_experts, d_ff_block, d_model) * 0.02 
        weights[f"blocks.{i}.moe.w2_weight"] = torch.randn(num_experts, d_model, d_ff_block) * 0.02 

    mtp_weights = {
        "mtp.embedding.weight": shared_weight.clone(),
        "mtp.head.weight": shared_weight.clone(),
        "mtp.proj_h.weight": torch.randn(d_model, d_model) * 0.02,
        "mtp.proj_emb.weight": torch.randn(d_model, d_model) * 0.02,
        "mtp.ln_fusion.weight": torch.ones(d_model),
        "mtp.ln_fusion.bias": torch.zeros(d_model),
        "mtp.norm1.weight": torch.ones(d_model),
        "mtp.norm1.bias": torch.zeros(d_model),
        "mtp.attn_qkv.weight": torch.randn(3 * d_model, d_model) * 0.02,
        "mtp.attn_qkv.bias": torch.zeros(3 * d_model),
        "mtp.attn_out.weight": torch.randn(d_model, d_model) * 0.02,
        "mtp.attn_out.bias": torch.zeros(d_model),
        "mtp.norm2.weight": torch.ones(d_model),
        "mtp.norm2.bias": torch.zeros(d_model),
        "mtp.mlp_fc1.weight": torch.randn(d_ff_mtp, d_model) * 0.02,
        "mtp.mlp_fc1.bias": torch.zeros(d_ff_mtp),
        "mtp.mlp_fc2.weight": torch.randn(d_model, d_ff_mtp) * 0.02,
        "mtp.mlp_fc2.bias": torch.zeros(d_model),
    }
    weights.update(mtp_weights)
    return weights