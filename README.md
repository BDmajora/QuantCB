# QuantCB: Latent Attention & Rust-Accelerated INT8 Engine

QuantCB is a high-performance Transformer engine optimized for "Local-First" environments. By combining **Multi-Head Latent Attention (MLA)** with a **Rust-backed BPE tokenizer** and **Symmetric INT8 Quantization**, this project bridges low-level systems engineering with state-of-the-art LLM architectures.

## Recent Milestone: MLA & Rust Integration
* **Implemented Multi-Head Latent Attention (MLA):** Inspired by DeepSeek-V3/FlashMLA. Successfully reduced KV Cache memory footprint by compressing Key-Value pairs into a low-rank latent vector bottleneck ($d_{latent}=128$).
* **Rust-Accelerated Tokenization:** Migrated core BPE training and encoding logic to Rust (`quantcb_rust`) using `PyO3`, achieving massive speedups in dataset preprocessing compared to pure Python implementations.

## Features

* **Multi-Head Latent Attention (MLA):** Significant reduction in VRAM/RAM usage during long-context generation. Decouples content and latent signaling via low-rank compression for superior scaling.
* **Hybrid Inference Engine:** Interactive runner allowing seamless switching between **FP32 (Base)** and **INT8 (Optimized)** models at runtime.
* **Symmetric INT8 Quantization:** Custom Post-Training Quantization (PTQ) pipeline. Reduces model footprint by **~2x** (e.g., 68.52 MB to 33.11 MB) while maintaining numerical integrity via a dequantize-on-load mechanism.
* **Rust BPE Core:** High-concurrency Byte-Pair Encoding with UTF-8 fallback. Ensures zero Out-of-Vocabulary (OOV) errors and rapid training on large technical corpora.
* **Performance Tracking:** Established baseline for small-scale training (Baseline: **Step 500 @ 4.7223 loss**).

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

* **To Train:** python3 run.py train
* **To Quantize:** python3 run.py optimize
* **To Run Inference:** python3 run.py inference
* **To Run Tests:** python3 run_tests.py

## Validation Suite

Located in /tests, these scripts ensure architectural integrity:
- **test_causal.py**: Validates that future tokens do not leak into the past.
- **test_attention.py**: Verifies the latent vector compression in the MLA block.
- **test_tokenizer.py**: Integrity suite for round-trip BPE encoding.