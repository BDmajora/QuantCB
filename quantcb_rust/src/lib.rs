use pyo3::prelude::*;

mod types;
mod bpe;
pub mod rope;      
pub mod attention; 
pub mod moe;       
pub mod layers;    
pub mod mtp_module; 
pub mod quantcb_model; 
pub mod ouro_engine; 

#[pymodule]
fn quantcb_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // --- BPE Functions ---
    m.add_function(wrap_pyfunction!(bpe::train_bpe, m)?)?;
    m.add_function(wrap_pyfunction!(bpe::encode_bpe, m)?)?;
    m.add_function(wrap_pyfunction!(bpe::decode_bpe, m)?)?;

    // --- QuantCB Model ---
    // Registered from the quantcb_model module
    m.add_class::<quantcb_model::QuantCbModel>()?;

    // --- Ouro Engine ---
    // Registered from the ouro_engine module
    m.add_class::<ouro_engine::OuroEngine>()?;

    // --- RoPE Scaling ---
    // Registered from the rope module
    m.add_class::<rope::DynamicNTKRotaryEmbedding>()?;

    Ok(())
}