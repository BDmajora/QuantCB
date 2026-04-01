use candle_core::{Result, Tensor, D};
use candle_nn::{linear_no_bias, Linear, Module, VarBuilder};

// Assuming QuantCB_FFN has been ported similarly 
// This maps to your `from models.layers import QuantCB_FFN`
use crate::layers::QuantCbFfn;

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
            // Give each expert its own VarBuilder subspace
            experts.push(QuantCbFfn::new(d_model, d_ff, vb.pp(format!("experts.{}", i)))?);
        }

        Ok(Self {
            num_experts,
            top_k,
            router,
            experts,
        })
    }

    pub fn forward(&mut self, x: &Tensor) -> Result<Tensor> {
        let (batch, seq_len, d_model) = x.dims3()?;
        let x_flat = x.reshape((batch * seq_len, d_model))?;

        // 1. Get Router Scores
        let router_logits = self.router.forward(&x_flat)?;
        let weights = candle_nn::ops::softmax(&router_logits, D::Minus1)?;

        // 2. Select Top-K Experts
        // arg_sort_last_dim(false) sorts in descending order to get the highest probabilities
        let sorted_indices = weights.arg_sort_last_dim(false)?;
        let top_k_indices = sorted_indices.narrow(D::Minus1, 0, self.top_k)?;
        
        // Gather the weights corresponding to the top-k indices
        let top_k_weights = weights.gather(&top_k_indices, D::Minus1)?;

        // Normalize weights so they sum to 1
        let sum_weights = top_k_weights.sum_keepdim(D::Minus1)?;
        let top_k_weights = top_k_weights.broadcast_div(&sum_weights)?;

        // 3. Dispatch and Aggregate
        let mut out = x_flat.zeros_like()?;

        // Pulling routing indices to CPU is the standard way to dynamically partition 
        // tokens in Candle without custom fused GPU kernels.
        let top_k_indices_cpu = top_k_indices.to_vec2::<u32>()?;
        
        // Flatten weights so we can 1D-index them precisely using (token_idx, expert_rank)
        let flat_weights = top_k_weights.flatten_all()?;

        for i in 0..self.num_experts {
            let mut tokens_for_expert = Vec::new();
            let mut ranks_for_expert = Vec::new();

            // Find which tokens in the batch were assigned to expert 'i'
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

            let num_tokens = tokens_for_expert.len();
            let token_idx_tensor = Tensor::from_vec(tokens_for_expert.clone(), num_tokens, x.device())?;

            // Get the tokens assigned to this expert
            let expert_input = x_flat.index_select(&token_idx_tensor, 0)?;

            // Run the expert
            let expert_out = self.experts[i].forward(&expert_input)?;

            // Extract the corresponding router weights for these tokens.
            // Using flat indexing to get exactly `top_k_weights[token_indices, expert_rank]`
            let mut flat_indices = Vec::with_capacity(num_tokens);
            for (t_idx, r_idx) in tokens_for_expert.iter().zip(ranks_for_expert.iter()) {
                flat_indices.push(t_idx * (self.top_k as u32) + r_idx);
            }
            let flat_idx_tensor = Tensor::from_vec(flat_indices, num_tokens, x.device())?;
            
            // Extract and unsqueeze so it broadcasts over d_model
            let extracted_weights = flat_weights
                .index_select(&flat_idx_tensor, 0)?
                .unsqueeze(1)?;

            // Multiply by the router weight
            let weighted_out = expert_out.broadcast_mul(&extracted_weights)?;

            // Add to the final output buffer mapping back to the token indices
            out = out.index_add(&token_idx_tensor, &weighted_out, 0)?;
        }

        // Reshape back to original dimensions
        out.reshape((batch, seq_len, d_model))
    }
}