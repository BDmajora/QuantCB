# QuantCB: Latent Attention & INT8 Inference Engine

QuantCB is a 16.68M parameter Transformer engine optimized for high-efficiency sequence modeling in CPU-constrained and "Local-First" environments. By implementing **Multi-Head Latent Attention (MLA)** and a custom **Symmetric INT8 Quantization** pipeline, this project bridges systems-level performance with modern LLM architecture.

## Features

* **Multi-Head Latent Attention (MLA):** Inspired by DeepSeek-V3/FlashMLA. Uses a latent vector bottleneck to compress Key-Value pairs, significantly reducing KV cache memory usage during long-context generation.
* **Inference Optimization (INT8 PTQ):** Features a custom Post-Training Quantization pipeline. Manually maps FP32 weights to INT8 precision using symmetric affine quantization, reducing the model footprint from **68.52 MB** to **33.11 MB** (**2.07x compression**).
* **Custom BPE Tokenizer:** A robust Byte-Pair Encoding implementation with a UTF-8 byte-fallback mechanism. Designed for specialized technical datasets with zero Out-of-Vocabulary (OOV) errors.
* **Decoupled Architecture:** Utilizes a Pre-Norm configuration with GELU activations and `torch.einsum` optimized matrix contractions for superior numerical stability.
* **Local-First Workflow:** Designed for offline deployment using local hardware (Ollama/Jan compatible philosophies) without unnecessary dependencies.

## Technical Stack

* **Frameworks:** Python 3.10+, PyTorch (2.0+), NumPy
* **Core Logic:** Multi-Head Latent Attention (MLA)
* **Optimization:** Symmetric Static INT8 Quantization
* **Environment:** Linux-first (Ubuntu/Debian), Venv-isolated

## Project Structure

* **models/attention.py**: Masked Multi-Head Latent Attention implementation.
* **models/layers.py**: FFN, RMSNorm, and Transformer Block logic.
* **models/quantcb_model.py**: Full Transformer wrapper and LM head.
* **src/tokenizer_basic.py**: Core logic for BPE training, encoding, and decoding.
* **src/train.py**: Structural verification loop and FP32 checkpointing.
* **src/optimize.py**: Quantization scripts for FP32 to INT8 weight conversion.
* **src/inference_int8.py**: Dequantization logic and inference verification.
* **modelOutput/**: Storage for `.pth` model checkpoints (Git ignored).

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