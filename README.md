# QuantCB: Latent Attention, Sparse MoE & FP8 Scaling Engine

QuantCB is a high-performance Transformer engine optimized for "Local-First" environments. By combining Multi-Head Latent Attention (MLA), Sparse Mixture-of-Experts (MoE), and a Rust-backed BPE tokenizer, this project bridges low-level systems engineering with state-of-the-art LLM architectures.

## Features

* **Multi-Token Prediction (MTP)**: Inspired by DeepSeek-V3. During training, the model uses an independent transformer-based module to predict the $t+2$ token. This "look-ahead" objective forces the base model to build more robust latent representations without adding computational overhead during inference.
* **Fine-Grained FP8 Quantization**: Advanced Post-Training Quantization (PTQ) pipeline. Utilizes E4M3 precision with block-level scaling factors to mitigate outlier-driven precision loss in MoE routers and experts.
* **Sparse Mixture-of-Experts (MoE)**: High-capacity model architecture with sparse activation. Allows the model to learn specialized tasks (syntax, punctuation, character names) across different expert sub-networks.
* **Multi-Head Latent Attention (MLA)**: Significant reduction in VRAM/RAM usage. Decouples content and latent signaling via low-rank compression for superior scaling during long-context generation.
* **DeepGEMM-Inspired Inference**: Interactive runner allowing seamless switching between FP32 (Base) and Optimized FP8 models with group-wise dequantization logic.
* **Rust BPE Core**: High-concurrency Byte-Pair Encoding with UTF-8 fallback. Ensures zero Out-of-Vocabulary (OOV) errors and rapid training on Shakespearean or technical corpora.
* **Performance Tracking**: Established MTP baseline (Baseline: Step 500 @ 5.2468 total loss with 8 experts and $t+2$ auxiliary signal).
## Technical Stack

* **Languages:** Python 3.10+, Rust (Edition 2021)
* **Frameworks:** PyTorch 2.0+, PyO3 (Python-Rust Bindings)
* **Core Logic:** Multi-Head Latent Attention (MLA)
* **Optimization:** Symmetric Static INT8 Quantization
* **Build Tools:** Cargo (Rust), Pip/Venv (Python)

## Project Structure

* **models/**:
    * `attention.py`: Masked MLA implementation with latent compression logic.
    * `layers.py`: FFN, LayerNorm, and Transformer Block logic.
    * `quantcb_model.py`: Full model wrapper with optimized KV-cache generation.
* **src/**:
    * `train.py`: Structural verification loop and FP32 checkpointing.
    * `generate.py`: Interactive inference script (FP32/INT8 selection).
    * `optimize.py`: Quantization pipeline for weight compression.
    * `tokenizer_basic.py`: Python wrapper for the Rust BPE engine.
* **quantcb_rust/**:
    * `src/lib.rs`: High-performance Rust implementation of BPE training and encoding.
* **modelOutput/**: Storage for `.pth` model checkpoints, `tokenizer.json`, and training data (Git ignored).

## Performance Tracking (CPU)

*Measured on a standard consumer-grade processor.*

| Metric | FP32 (Base) | INT8 (Optimized) |
| :--- | :--- | :--- |
| Model Size | 68.52 MB | 33.11 MB |
| Compression | 1.0x | 2.07x |
| Status | Verified | Verified |

## Installation

1. Clone the repository:
   git clone https://github.com/BDmajora/QuantCB.git
   cd QuantCB

2. Install the Rust Toolchain (Required for BPE training):
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   . "$HOME/.cargo/env"

3. Set up a virtual environment:
   python3 -m venv venv
   source venv/bin/activate

4. Install dependencies and compile the Rust extension:
   pip install -r requirements.txt
   cd quantcb_rust
   maturin develop --release
   cd ..

## Usage

The project uses a root-level runner for all operations:

* **To Run Trainer or Generate:** python3 run.py
* **To Run Tests:** python3 run_tests.py