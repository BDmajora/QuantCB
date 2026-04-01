use candle_core::{Device, Result, Tensor, D};
use candle_nn::{linear_no_bias, Linear, Module, VarBuilder};
use pyo3::prelude::*;

use crate::quantcb_model::QuantCbModel;

#[pyclass]
pub struct OuroEngine {
    pub model: QuantCbModel,
    pub max_loops: usize,
    pub exit_threshold: f64,
    pub latent_probe: Linear,
    pub thinking_gate: Tensor, 
}

impl OuroEngine {
    pub fn new(
        model: QuantCbModel,
        max_loops: usize,
        exit_threshold: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        let d_model = model.d_model;
        
        let latent_probe = linear_no_bias(d_model, 1, vb.pp("latent_probe"))?;
        
        let thinking_gate = vb.get_with_hints(
            (1,), 
            "thinking_gate", 
            candle_nn::Init::Const(0.0)
        )?;

        Ok(Self {
            model,
            max_loops,
            exit_threshold,
            latent_probe,
            thinking_gate,
        })
    }

    pub fn forward(
        &mut self,
        idx: &Tensor,
        targets: Option<&Tensor>,
        mask: Option<&Tensor>,
        past_key_values: Option<&Vec<Tensor>>,
    ) -> Result<(Tensor, Option<Tensor>, Vec<Tensor>)> {
        let (_batch_size, seq_len) = idx.dims2()?;
        let device = idx.device();

        let causal_mask = if mask.is_none() && seq_len > 1 && past_key_values.is_none() {
            Some(self.get_causal_mask(seq_len, device)?)
        } else {
            mask.cloned()
        };

        let mut loop_x = self.model.token_embedding.forward(idx)?;
        let mut final_logits = loop_x.clone(); 
        let mut final_presents = Vec::new();
        let mut total_aux_loss = Tensor::new(0f32, device)?;

        for loop_idx in 0..self.max_loops {
            let prev_x = loop_x.clone();
            let mut current_loop_aux = Tensor::new(0f32, device)?;
            let mut current_loop_presents = Vec::new();

            let gate = candle_nn::ops::sigmoid(&self.thinking_gate)?;
            
            let mut current_x = loop_x;
            for (i, block) in self.model.blocks.iter_mut().enumerate() {
                let layer_past = past_key_values.and_then(|p| p.get(i));
                
                // Explicit type annotation ensures the compiler knows what block.forward returns
                let (block_out, present, l_aux): (Tensor, Option<Tensor>, Tensor) = 
                    block.forward(&current_x, causal_mask.as_ref(), layer_past)?;
                
                current_x = block_out;
                current_loop_aux = current_loop_aux.add(&l_aux)?;

                // Cache KV pairs during every loop so they are available if we exit early
                if let Some(p) = present {
                    current_loop_presents.push(p);
                }
            }

            let gate_inv = (Tensor::new(1.0f32, device)? - &gate)?;
            loop_x = prev_x.broadcast_mul(&gate_inv)?.add(&current_x.broadcast_mul(&gate)?)?;

            let h_n = self.model.ln_f.forward(&loop_x)?;
            final_logits = self.model.lm_head.forward(&h_n)?;
            
            if loop_idx == self.max_loops - 1 {
                // .clone() prevents moving the vector here so it can be checked in early-exit
                final_presents = current_loop_presents.clone();
                total_aux_loss = current_loop_aux;
            }

            if targets.is_none() && seq_len == 1 {
                if self.should_exit_early(&final_logits, device)? {
                    // Safe to move the vector here because we immediately break out of the loop
                    final_presents = current_loop_presents; 
                    break; 
                }
            }
        }

        let mut loss_out = None;
        if let Some(t) = targets {
            let loss_main = candle_nn::loss::cross_entropy(
                &final_logits.flatten_to(D::Minus2)?, 
                &t.flatten_all()?
            )?;
            let scaled_aux = total_aux_loss.affine(0.01, 0.0)?;
            loss_out = Some(loss_main.add(&scaled_aux)?);
        }

        Ok((final_logits, loss_out, final_presents))
    }

    fn should_exit_early(&self, logits: &Tensor, device: &Device) -> Result<bool> {
        let last_logits = logits.narrow(1, logits.dim(1)? - 1, 1)?.squeeze(1)?;
        let probs = candle_nn::ops::softmax(&last_logits, D::Minus1)?;
        let log_probs = probs.add(&Tensor::new(1e-9f32, device)?)?.log()?;
        let entropy = probs.mul(&log_probs)?.sum_all()?.neg()?;
        let entropy_val = entropy.to_vec0::<f32>()?;
        Ok((entropy_val as f64) < self.exit_threshold)
    }

    fn get_causal_mask(&self, seq_len: usize, device: &Device) -> Result<Tensor> {
        let mask: Vec<_> = (0..seq_len)
            .flat_map(|i| (0..seq_len).map(move |j| if j > i { f32::NEG_INFINITY } else { 0.0 }))
            .collect();
        Tensor::from_vec(mask, (1, 1, seq_len, seq_len), device)
    }
}