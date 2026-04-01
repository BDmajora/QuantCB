use candle_core::{Result, Tensor, D};
use candle_nn::{layer_norm, linear_no_bias, LayerNorm, Linear, Module, VarBuilder};

use crate::rope::{rotate_half, DynamicNTKRotaryEmbedding};

fn apply_rope_single(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let x_rotated = rotate_half(x)?;
    let x_cos = x.broadcast_mul(cos)?;
    let x_sin_rotated = x_rotated.broadcast_mul(sin)?;
    x_cos.add(&x_sin_rotated)
}

pub struct MlaAttention {
    n_heads: usize,
    head_dim: usize,
    qk_rope_dim: usize,
    qk_nope_dim: usize,
    w_q: Linear,
    w_dkv: Linear,
    ln_kv: LayerNorm,
    w_uk: Linear,
    w_uv: Linear,
    w_o: Linear,
    rope: DynamicNTKRotaryEmbedding,
}

impl MlaAttention {
    pub fn new(
        d_model: usize,
        n_heads: usize,
        latent_dim: usize,
        head_dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let qk_rope_dim = head_dim / 2;
        let qk_nope_dim = head_dim - qk_rope_dim;

        let w_q = linear_no_bias(d_model, n_heads * head_dim, vb.pp("W_q"))?;
        let w_dkv = linear_no_bias(d_model, latent_dim, vb.pp("W_dkv"))?;
        let ln_kv = layer_norm(latent_dim, 1e-5, vb.pp("ln_kv"))?;
        let w_uk = linear_no_bias(latent_dim, n_heads * head_dim, vb.pp("W_uk"))?;
        let w_uv = linear_no_bias(latent_dim, n_heads * head_dim, vb.pp("W_uv"))?;
        let w_o = linear_no_bias(n_heads * head_dim, d_model, vb.pp("W_o"))?;

        let rope = DynamicNTKRotaryEmbedding::new(qk_rope_dim, 256, 10000.0, vb.device())?;

        Ok(Self {
            n_heads,
            head_dim,
            qk_rope_dim,
            qk_nope_dim,
            w_q,
            w_dkv,
            ln_kv,
            w_uk,
            w_uv,
            w_o,
            rope,
        })
    }

    pub fn forward(
        &mut self,
        x: &Tensor,
        mask: Option<&Tensor>,
        layer_past: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let (batch_size, seq_len, _) = x.dims3()?;

        let q = self.w_q.forward(x)?.reshape((batch_size, seq_len, self.n_heads, self.head_dim))?;
        let q_nope = q.narrow(D::Minus1, 0, self.qk_nope_dim)?.transpose(1, 2)?;
        let q_pe = q.narrow(D::Minus1, self.qk_nope_dim, self.qk_rope_dim)?.transpose(1, 2)?;

        let compressed = self.w_dkv.forward(x)?;
        let c_kv = self.ln_kv.forward(&compressed)?;

        let present_c_kv = match layer_past {
            Some(past) => Tensor::cat(&[past, &c_kv], 1)?,
            None => c_kv,
        };

        let full_seq_len = present_c_kv.dim(1)?;
        let k = self.w_uk.forward(&present_c_kv)?
            .reshape((batch_size, full_seq_len, self.n_heads, self.head_dim))?;
        let v = self.w_uv.forward(&present_c_kv)?
            .reshape((batch_size, full_seq_len, self.n_heads, self.head_dim))?
            .transpose(1, 2)?;

        let k_nope = k.narrow(D::Minus1, 0, self.qk_nope_dim)?.transpose(1, 2)?;
        let k_pe = k.narrow(D::Minus1, self.qk_nope_dim, self.qk_rope_dim)?.transpose(1, 2)?;

        let (cos, sin) = self.rope.forward(full_seq_len)?;

        let cos_q = cos.narrow(2, full_seq_len - seq_len, seq_len)?;
        let sin_q = sin.narrow(2, full_seq_len - seq_len, seq_len)?;

        let q_pe = apply_rope_single(&q_pe, &cos_q, &sin_q)?;
        let k_pe = apply_rope_single(&k_pe, &cos, &sin)?;

        let q_final = Tensor::cat(&[q_nope, q_pe], D::Minus1)?;
        let k_final = Tensor::cat(&[k_nope, k_pe], D::Minus1)?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut attn_scores = (q_final.matmul(&k_final.transpose(2, 3)?)? * scale)?;

        if let Some(m) = mask {
            attn_scores = attn_scores.broadcast_add(m)?;
        }

        let attn_weights = candle_nn::ops::softmax(&attn_scores, D::Minus1)?;
        let context = attn_weights.matmul(&v)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch_size, seq_len, self.n_heads * self.head_dim))?;

        Ok((self.w_o.forward(&context)?, present_c_kv))
    }
}