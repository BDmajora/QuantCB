use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{embedding, linear_no_bias, Embedding, Linear, Module, RmsNorm, VarBuilder, rms_norm, VarMap};
use pyo3::prelude::*;

use crate::layers::QuantCbBlock;
use crate::mtp_module::MtpModule;

#[pyclass]
pub struct QuantCbModel {
    pub token_embedding: Embedding,
    pub blocks: Vec<QuantCbBlock>,
    pub ln_f: RmsNorm,
    pub lm_head: Linear,
    pub mtp: MtpModule,
    pub latent_probe: Linear,
    pub d_model: usize,
}

impl QuantCbModel {
    pub fn new(
        vocab_size: usize,
        d_model: usize,
        n_heads: usize,
        d_ff: usize,
        n_layers: usize,
        latent_dim: usize,
        head_dim: usize,
        num_experts: usize,
        top_k: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let token_embedding = embedding(vocab_size, d_model, vb.pp("token_embedding"))?;
        
        let mut blocks = Vec::with_capacity(n_layers);
        let blocks_vb = vb.pp("blocks");
        for i in 0..n_layers {
            blocks.push(QuantCbBlock::new(
                d_model,
                n_heads,
                d_ff,
                latent_dim,
                head_dim,
                num_experts,
                top_k,
                blocks_vb.pp(i),
            )?);
        }

        let ln_f = rms_norm(d_model, 1e-6, vb.pp("ln_f"))?;
        let head_weight = token_embedding.embeddings().clone();
        let lm_head = Linear::new(head_weight, None);
        let mtp = MtpModule::new(d_model, n_heads, d_ff, vb.pp("mtp"))?;
        let latent_probe = linear_no_bias(d_model, 1, vb.pp("latent_probe"))?;

        Ok(Self {
            token_embedding,
            blocks,
            ln_f,
            lm_head,
            mtp,
            latent_probe,
            d_model,
        })
    }

    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let mut x = self.token_embedding.forward(input_ids)?;
        let mut total_l_aux = Tensor::new(0f32, x.device())?.to_dtype(x.dtype())?;

        for block in self.blocks.iter_mut() {
            // FIXED: Explicit type annotation for the destructuring
            let (block_out, _present, l_aux): (Tensor, Option<Tensor>, Tensor) = 
                block.forward(&x, mask, None)?;
            
            x = block_out;
            total_l_aux = total_l_aux.add(&l_aux)?;
        }

        let h_n = self.ln_f.forward(&x)?;
        let logits = self.lm_head.forward(&h_n)?;

        Ok((logits, h_n, total_l_aux))
    }

    pub fn get_hallucination_score(&self, h_n: &Tensor) -> Result<Tensor> {
        let score = self.latent_probe.forward(h_n)?;
        candle_nn::ops::sigmoid(&score)
    }

    pub fn predict_mtp(
        &mut self,
        h_base: &Tensor,
        targets: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        self.mtp.forward(h_base, targets, &self.token_embedding, &self.lm_head)
    }
}

#[pymethods]
impl QuantCbModel {
    #[staticmethod]
    pub fn load(path: String, vocab_size: usize, d_model: usize) -> PyResult<Self> {
        let device = Device::Cpu;
        let mut varmap = VarMap::new();
        varmap.load(path).map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let model = Self::new(vocab_size, d_model, 8, 1024, 6, 128, 64, 8, 2, vb)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(model)
    }
}