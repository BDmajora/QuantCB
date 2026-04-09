import torch
from config import *

class ParamContainer(torch.nn.Module):
    def __init__(self, n_layers, vocab_size, d_model, n_heads, head_dim, latent_dim, num_experts):
        super().__init__()
        self.n_layers = n_layers
        
        # 1. Global / Embedding Weights
        self.weights = torch.nn.ParameterDict({
            "token_embedding_DOT_weight": torch.nn.Parameter(torch.randn(vocab_size, d_model)),
            "ln_f_DOT_weight": torch.nn.Parameter(torch.ones(d_model)),
            "lm_head_DOT_weight": torch.nn.Parameter(torch.randn(vocab_size, d_model)),
        })
        
        # 2. MTP (Multi-Token Prediction) Module Weights
        self.weights.update({
            "mtp_DOT_embedding_weight": torch.nn.Parameter(torch.randn(vocab_size, d_model)),
            "mtp_DOT_head_weight": torch.nn.Parameter(torch.randn(vocab_size, d_model)),
            "mtp_DOT_proj_h_weight": torch.nn.Parameter(torch.randn(d_model, d_model)),
            "mtp_DOT_proj_emb_weight": torch.nn.Parameter(torch.randn(d_model, d_model)),
            "mtp_DOT_ln_fusion_weight": torch.nn.Parameter(torch.ones(d_model)),
            "mtp_DOT_ln_fusion_bias": torch.nn.Parameter(torch.zeros(d_model)),
            "mtp_DOT_norm1_weight": torch.nn.Parameter(torch.ones(d_model)),
            "mtp_DOT_norm1_bias": torch.nn.Parameter(torch.zeros(d_model)),
            "mtp_DOT_attn_qkv_weight": torch.nn.Parameter(torch.randn(3 * d_model, d_model)),
            "mtp_DOT_attn_qkv_bias": torch.nn.Parameter(torch.zeros(3 * d_model)),
            "mtp_DOT_attn_out_weight": torch.nn.Parameter(torch.randn(d_model, d_model)),
            "mtp_DOT_attn_out_bias": torch.nn.Parameter(torch.zeros(d_model)),
            "mtp_DOT_norm2_weight": torch.nn.Parameter(torch.ones(d_model)),
            "mtp_DOT_norm2_bias": torch.nn.Parameter(torch.zeros(d_model)),
            "mtp_DOT_mlp_fc1_weight": torch.nn.Parameter(torch.randn(4 * d_model, d_model)),
            "mtp_DOT_mlp_fc1_bias": torch.nn.Parameter(torch.zeros(4 * d_model)),
            "mtp_DOT_mlp_fc2_weight": torch.nn.Parameter(torch.randn(d_model, 4 * d_model)),
            "mtp_DOT_mlp_fc2_bias": torch.nn.Parameter(torch.zeros(d_model)),
        })

        # 3. Per-Block Weights (MLA + MoE)
        for i in range(n_layers):
            self.weights[f"blocks_DOT_{i}_DOT_ln1_DOT_weight"] = torch.nn.Parameter(torch.ones(d_model))
            self.weights[f"blocks_DOT_{i}_DOT_ln2_DOT_weight"] = torch.nn.Parameter(torch.ones(d_model))
            
            pfx = f"blocks_DOT_{i}_DOT_attn_DOT_"
            self.weights[pfx + "wq_weight"] = torch.nn.Parameter(torch.randn(n_heads * head_dim, d_model))
            self.weights[pfx + "wdkv_weight"] = torch.nn.Parameter(torch.randn(latent_dim, d_model))
            self.weights[pfx + "ln_kv_weight"] = torch.nn.Parameter(torch.ones(latent_dim))
            self.weights[pfx + "ln_kv_bias"] = torch.nn.Parameter(torch.zeros(latent_dim))
            self.weights[pfx + "wuk_weight"] = torch.nn.Parameter(torch.randn(n_heads * head_dim, latent_dim))
            self.weights[pfx + "wuv_weight"] = torch.nn.Parameter(torch.randn(n_heads * head_dim, latent_dim))
            self.weights[pfx + "wo_weight"] = torch.nn.Parameter(torch.randn(d_model, n_heads * head_dim))
            self.weights[pfx + "rope_inv_freq"] = torch.nn.Parameter(torch.randn(head_dim // 2))

            mfx = f"blocks_DOT_{i}_DOT_moe_DOT_"
            expert_dim = 4 * d_model 
            self.weights[mfx + "router_weight"] = torch.nn.Parameter(torch.randn(num_experts, d_model))
            self.weights[mfx + "w1_weight"] = torch.nn.Parameter(torch.randn(num_experts, expert_dim, d_model))
            self.weights[mfx + "w2_weight"] = torch.nn.Parameter(torch.randn(num_experts, d_model, expert_dim))
            self.weights[mfx + "w3_weight"] = torch.nn.Parameter(torch.randn(num_experts, expert_dim, d_model))

    def get_stateless_dict(self):
        """Converts _DOT_ keys to real dots for the model's weight.get() calls."""
        return {k.replace("_DOT_", "."): v for k, v in self.weights.items()}