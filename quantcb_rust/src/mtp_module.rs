use candle_core::{Device, Result, Tensor, D};
use candle_nn::{layer_norm, linear_no_bias, Embedding, LayerNorm, Linear, Module, VarBuilder};

// Importing our previous FFN; assuming standard MultiHeadAttention for the mixer
// or you can swap this with the MlaAttention from the previous step.
use crate::layers::QuantCbFfn;

/// A standard Transformer Encoder Block (Pre-norm) to act as the "Mixer"
pub struct MixerBlock {
    attn_norm: LayerNorm,
    // Note: Using a standard MHA here as is typical for MTP mixers
    // but you can replace with MlaAttention if your architecture demands it.
    w_qkv: Linear, 
    w_out: Linear,
    ffn_norm: LayerNorm,
    ffn: QuantCbFfn,
    n_heads: usize,
    head_dim: usize,
}

impl MixerBlock {
    pub fn new(d_model: usize, n_heads: usize, d_ff: usize, vb: VarBuilder) -> Result<Self> {
        let head_dim = d_model / n_heads;
        let attn_norm = layer_norm(d_model, 1e-5, vb.pp("attn_norm"))?;
        let w_qkv = linear_no_bias(d_model, 3 * d_model, vb.pp("w_qkv"))?;
        let w_out = linear_no_bias(d_model, d_model, vb.pp("w_out"))?;
        let ffn_norm = layer_norm(d_model, 1e-5, vb.pp("ffn_norm"))?;
        let ffn = QuantCbFfn::new(d_model, d_ff, vb.pp("ffn"))?;

        Ok(Self {
            attn_norm,
            w_qkv,
            w_out,
            ffn_norm,
            ffn,
            n_heads,
            head_dim,
        })
    }

    pub fn forward(&mut self, x: &Tensor, mask: &Tensor) -> Result<Tensor> {
        // 1. Attention Path (Pre-Norm)
        let residual = x;
        let x = self.attn_norm.forward(x)?;
        let (b, s, d) = x.dims3()?;
        
        let qkv = self.w_qkv.forward(&x)?;
        let qkv = qkv.reshape((b, s, 3, self.n_heads, self.head_dim))?.transpose(1, 3)?;
        let q = qkv.narrow(2, 0, 1)?.squeeze(2)?;
        let k = qkv.narrow(2, 1, 1)?.squeeze(2)?;
        let v = qkv.narrow(2, 2, 1)?.squeeze(2)?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        let scores = scores.broadcast_add(mask)?;
        let probs = candle_nn::ops::softmax(&scores, D::Minus1)?;
        let attn_out = probs.matmul(&v)?;
        
        let attn_out = attn_out
            .transpose(1, 2)?
            .reshape((b, s, d))?;
        let attn_out = self.w_out.forward(&attn_out)?;
        let x = (residual + attn_out)?;

        // 2. FFN Path (Pre-Norm)
        let residual = &x;
        let x = self.ffn_norm.forward(&x)?;
        let x = self.ffn.forward(&x)?;
        x.add(residual)
    }
}

pub struct MtpModule {
    proj_h: Linear,
    proj_emb: Linear,
    ln_fusion: LayerNorm,
    mixer: MixerBlock,
}

impl MtpModule {
    pub fn new(
        d_model: usize,
        n_heads: usize,
        d_ff: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        // Initialize with small std as per DeepSeek-style (0.01)
        // In Candle, we use VarBuilder's initialization configuration if needed,
        // but here we define the layers.
        let proj_h = linear_no_bias(d_model, d_model, vb.pp("proj_h"))?;
        let proj_emb = linear_no_bias(d_model, d_model, vb.pp("proj_emb"))?;
        let ln_fusion = layer_norm(d_model, 1e-5, vb.pp("ln_fusion"))?;
        
        let mixer = MixerBlock::new(d_model, n_heads, d_ff, vb.pp("mixer"))?;

        Ok(Self {
            proj_h,
            proj_emb,
            ln_fusion,
            mixer,
        })
    }

    /// Generates a causal mask for the transformer mixer
    fn get_causal_mask(&self, seq_len: usize, device: &Device) -> Result<Tensor> {
        let mask: Vec<_> = (0..seq_len)
            .flat_map(|i| (0..seq_len).map(move |j| if j > i { f32::NEG_INFINITY } else { 0f32 }))
            .collect();
        Tensor::from_vec(mask, (seq_len, seq_len), device)
    }

    pub fn forward(
        &mut self,
        h_base: &Tensor,      // Hidden states from base model at t
        targets: &Tensor,     // Hint tokens at t+1
        embedding: &Embedding, // Shared from base
        head: &Linear,        // Shared from base
    ) -> Result<(Tensor, Tensor)> {
        // 1. Get embeddings for hint tokens (t+1)
        let x_embed = embedding.forward(targets)?;

        // 2. Additive Fusion (0.5 scale for variance stability)
        let h_proj = self.proj_h.forward(h_base)?;
        let emb_proj = self.proj_emb.forward(&x_embed)?;
        let fused = (h_proj.add(&emb_proj)? * 0.5)?;
        let x = self.ln_fusion.forward(&fused)?;

        // 3. Process through Causal Mixer
        let seq_len = x.dim(1)?;
        let mask = self.get_causal_mask(seq_len, x.device())?;
        let x_mtp = self.mixer.forward(&x, &mask)?;

        // 4. Predict t+2 using shared head
        let logits = head.forward(&x_mtp)?;

        Ok((logits, x_mtp))
    }
}