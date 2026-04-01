use candle_core::{DType, Device, Result, Tensor, D};
use pyo3::prelude::*;

#[pyclass]
pub struct DynamicNTKRotaryEmbedding {
    #[pyo3(get)]
    pub dim: usize,
    #[pyo3(get)]
    pub max_position_embeddings: usize,
    pub base: f64,
    pub max_seq_len_cached: usize,
    pub cos_cached: Tensor,
    pub sin_cached: Tensor,
    pub device: Device,
}

impl DynamicNTKRotaryEmbedding {
    pub fn new(
        dim: usize,
        max_position_embeddings: usize,
        base: f64,
        device: &Device,
    ) -> Result<Self> {
        let mut obj = Self {
            dim,
            max_position_embeddings,
            base,
            max_seq_len_cached: 0,
            cos_cached: Tensor::zeros((1, 1, 1, dim), DType::F32, device)?,
            sin_cached: Tensor::zeros((1, 1, 1, dim), DType::F32, device)?,
            device: device.clone(),
        };
        obj.update_cache(max_position_embeddings)?;
        Ok(obj)
    }

    fn update_cache(&mut self, seq_len: usize) -> Result<()> {
        self.max_seq_len_cached = seq_len;

        // Dynamic NTK Scaling logic
        let current_base = if seq_len > self.max_position_embeddings {
            let ratio = seq_len as f64 / self.max_position_embeddings as f64;
            let exponent = self.dim as f64 / (self.dim as f64 - 2.0);
            // standard NTK scaling: base * (ratio).powf(exponent)
            // Note: kept your (ratio - 0.5) logic if that's your specific implementation
            self.base * (ratio - 0.5).max(1.0).powf(exponent)
        } else {
            self.base
        };

        let inv_freq: Vec<f32> = (0..self.dim)
            .step_by(2)
            .map(|i| 1.0 / (current_base.powf(i as f64 / self.dim as f64)) as f32)
            .collect();
        let inv_freq = Tensor::from_vec(inv_freq, (1, self.dim / 2), &self.device)?;

        let t = Tensor::arange(0u32, seq_len as u32, &self.device)?
            .to_dtype(DType::F32)?
            .reshape((seq_len, 1))?;
        
        // freqs shape: (seq_len, dim/2)
        let freqs = t.broadcast_mul(&inv_freq)?;

        // emb shape: (seq_len, dim)
        let emb = Tensor::cat(&[&freqs, &freqs], D::Minus1)?;

        // Final cache shape: [1, 1, seq_len, dim]
        self.cos_cached = emb.cos()?.unsqueeze(0)?.unsqueeze(0)?;
        self.sin_cached = emb.sin()?.unsqueeze(0)?.unsqueeze(0)?;

        Ok(())
    }

    pub fn forward(&mut self, seq_len: usize) -> Result<(Tensor, Tensor)> {
        if seq_len > self.max_seq_len_cached {
            self.update_cache(seq_len)?;
        }

        let cos = self.cos_cached.narrow(2, 0, seq_len)?;
        let sin = self.sin_cached.narrow(2, 0, seq_len)?;

        Ok((cos, sin))
    }
}

// Internal Helper Functions
pub fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let last_dim = x.dim(D::Minus1)?;
    let x1 = x.narrow(D::Minus1, 0, last_dim / 2)?;
    let x2 = x.narrow(D::Minus1, last_dim / 2, last_dim / 2)?;
    let neg_x2 = x2.neg()?;
    Tensor::cat(&[&neg_x2, &x1], D::Minus1)
}

pub fn apply_rotary_pos_emb(
    q: &Tensor, 
    k: &Tensor, 
    cos: &Tensor, 
    sin: &Tensor
) -> Result<(Tensor, Tensor)> {
    let q_embed = (q.broadcast_mul(cos)? + rotate_half(q)?.broadcast_mul(sin)?)?;
    let k_embed = (k.broadcast_mul(cos)? + rotate_half(k)?.broadcast_mul(sin)?)?;
    Ok((q_embed, k_embed))
}