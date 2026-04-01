use candle_core::{Result, Tensor, D};
use candle_nn::{linear_no_bias, rms_norm, Linear, Module, RmsNorm, VarBuilder};

use crate::attention::MlaAttention;

/// QuantCB_FFN: Standard MLP with GELU
pub struct QuantCbFfn {
    w_1: Linear,
    w_2: Linear,
}

impl QuantCbFfn {
    pub fn new(d_model: usize, d_ff: usize, vb: VarBuilder) -> Result<Self> {
        // In Candle, standard practice is to handle initialization via the VarBuilder 
        // prior to passing it in, but the default init works well for general usage.
        let w_1 = linear_no_bias(d_model, d_ff, vb.pp("w_1"))?;
        let w_2 = linear_no_bias(d_ff, d_model, vb.pp("w_2"))?;

        Ok(Self { w_1, w_2 })
    }

    pub fn forward(&mut self, x: &Tensor) -> Result<Tensor> {
        // Dropout is omitted for inference. 
        // x -> w_1 -> gelu -> w_2
        let x = self.w_1.forward(x)?;
        let x = x.gelu()?;
        self.w_2.forward(&x)
    }
}

/// QuantCB_MoE: Mixture of Experts with Load Balancing Loss
pub struct QuantCbMoe {
    num_experts: usize,
    top_k: usize,
    router: Linear,
    experts: Vec<QuantCbFfn>,
}

impl QuantCbMoe {
    pub fn new(
        d_model: usize,
        d_ff: usize,
        num_experts: usize,
        top_k: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let router = linear_no_bias(d_model, num_experts, vb.pp("router"))?;
        
        let mut experts = Vec::with_capacity(num_experts);
        for i in 0..num_experts {
            experts.push(QuantCbFfn::new(d_model, d_ff, vb.pp(format!("experts.{}", i)))?);
        }

        Ok(Self {
            num_experts,
            top_k,
            router,
            experts,
        })
    }

    pub fn forward(&mut self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let (batch, seq_len, d_model) = x.dims3()?;
        let num_tokens = batch * seq_len;
        let x_flat = x.reshape((num_tokens, d_model))?;

        // 1. Router logits & probabilities
        let router_logits = self.router.forward(&x_flat)?;
        let weights = candle_nn::ops::softmax(&router_logits, D::Minus1)?;

        // 2. Select Top-K Experts
        let sorted_indices = weights.arg_sort_last_dim(false)?;
        let top_k_indices = sorted_indices.narrow(D::Minus1, 0, self.top_k)?;
        
        // Gather and normalize weights
        let top_k_weights = weights.gather(&top_k_indices, D::Minus1)?;
        let sum_weights = top_k_weights.sum_keepdim(D::Minus1)?;
        
        // Add 1e-6 to denominator as in the Python code
        let eps = Tensor::new(1e-6f32, x.device())?.to_dtype(x.dtype())?;
        let sum_weights_eps = sum_weights.broadcast_add(&eps)?;
        let top_k_weights = top_k_weights.broadcast_div(&sum_weights_eps)?;

        // Extract indices to CPU for routing & loss calculation
        let top_k_indices_cpu = top_k_indices.to_vec2::<u32>()?;

        // 3. Load Balancing Loss (l_aux)
        // mean_probs: (num_experts,)
        let mean_probs = weights.mean(0)?; 
        
        // Compute density_probs efficiently on CPU without massive one_hot tensors
        let mut expert_counts = vec![0.0f32; self.num_experts];
        let total_topk = (num_tokens * self.top_k) as f32;
        for ranks in top_k_indices_cpu.iter() {
            for &exp_idx in ranks.iter() {
                expert_counts[exp_idx as usize] += 1.0;
            }
        }
        
        let density_probs_vec: Vec<f32> = expert_counts.into_iter().map(|c| c / total_topk).collect();
        let density_probs = Tensor::from_vec(density_probs_vec, (self.num_experts,), x.device())?
            .to_dtype(x.dtype())?;
        
        // l_aux = num_experts * sum(mean_probs * density_probs)
        let l_aux_val = (mean_probs.mul(&density_probs)?.sum_all()? * (self.num_experts as f64))?;

        // 4. Dispatch and Aggregate
        let mut out = x_flat.zeros_like()?;
        let flat_weights = top_k_weights.flatten_all()?;

        for i in 0..self.num_experts {
            let mut tokens_for_expert = Vec::new();
            let mut ranks_for_expert = Vec::new();

            for (token_idx, ranks) in top_k_indices_cpu.iter().enumerate() {
                for (rank, &exp_idx) in ranks.iter().enumerate() {
                    if exp_idx as usize == i {
                        tokens_for_expert.push(token_idx as u32);
                        ranks_for_expert.push(rank as u32);
                    }
                }
            }

            if tokens_for_expert.is_empty() {
                continue;
            }

            let num_exp_tokens = tokens_for_expert.len();
            let token_idx_tensor = Tensor::from_vec(tokens_for_expert.clone(), num_exp_tokens, x.device())?;

            let expert_input = x_flat.index_select(&token_idx_tensor, 0)?;
            let expert_out = self.experts[i].forward(&expert_input)?;

            let mut flat_indices = Vec::with_capacity(num_exp_tokens);
            for (t_idx, r_idx) in tokens_for_expert.iter().zip(ranks_for_expert.iter()) {
                flat_indices.push(t_idx * (self.top_k as u32) + r_idx);
            }
            let flat_idx_tensor = Tensor::from_vec(flat_indices, num_exp_tokens, x.device())?;
            
            let extracted_weights = flat_weights
                .index_select(&flat_idx_tensor, 0)?
                .unsqueeze(1)?;

            let weighted_out = expert_out.broadcast_mul(&extracted_weights)?;
            out = out.index_add(&token_idx_tensor, &weighted_out, 0)?;
        }

        let out_reshaped = out.reshape((batch, seq_len, d_model))?;
        Ok((out_reshaped, l_aux_val))
    }
}

/// QuantCB_Block: Transformer block combining Attention and MoE
pub struct QuantCbBlock {
    ln_1: RmsNorm,
    attn: MlaAttention,
    ln_2: RmsNorm,
    moe: QuantCbMoe,
}

impl QuantCbBlock {
    pub fn new(
        d_model: usize,
        n_heads: usize,
        d_ff: usize,
        latent_dim: usize,
        head_dim: usize,
        num_experts: usize,
        top_k: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let ln_1 = rms_norm(d_model, 1e-6, vb.pp("ln_1"))?;
        let attn = MlaAttention::new(d_model, n_heads, latent_dim, head_dim, vb.pp("attn"))?;
        let ln_2 = rms_norm(d_model, 1e-6, vb.pp("ln_2"))?;
        let moe = QuantCbMoe::new(d_model, d_ff, num_experts, top_k, vb.pp("moe"))?;

        Ok(Self {
            ln_1,
            attn,
            ln_2,
            moe,
        })
    }

    pub fn forward(
        &mut self,
        x: &Tensor,
        mask: Option<&Tensor>,
        layer_past: Option<&Tensor>,
    ) -> Result<(Tensor, Option<Tensor>, Tensor)> {
        // 1. Attention Path
        let residual = x;
        let normed_x = self.ln_1.forward(x)?;
        
        let (attn_out, present) = self.attn.forward(&normed_x, mask, layer_past)?;
        let x_after_attn = residual.add(&attn_out)?;

        // 2. MoE Path
        let residual_moe = &x_after_attn;
        let normed_x_moe = self.ln_2.forward(&x_after_attn)?;
        
        let (moe_out, l_aux) = self.moe.forward(&normed_x_moe)?;
        let x_final = residual_moe.add(&moe_out)?;

        Ok((x_final, Some(present), l_aux))
    }
}