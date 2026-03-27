# QuantCB: Latent Attention, Sparse MoE & Rust-Accelerated Engine

QuantCB is a high-performance Transformer engine optimized for "Local-First" environments. By combining Multi-Head Latent Attention (MLA), Sparse Mixture-of-Experts (MoE), and a Rust-backed BPE tokenizer, this project bridges low-level systems engineering with state-of-the-art LLM architectures.

## Recent Milestone: MLA & MoE Integration
* Implemented Multi-Head Latent Attention (MLA): Inspired by DeepSeek-V3. Successfully reduced KV Cache memory footprint by compressing Key-Value pairs into a low-rank latent vector bottleneck ($d_{latent}=128$).
* Sparse Mixture-of-Experts (MoE): Integrated a 1-of-8 expert routing system. Achieved significant parameter scaling (8 specialized experts) while maintaining a constant CPU "Active" compute cost (top-k=2).
* Anti-Loop Inference Logic: Implemented Repetition Penalty and Top-P (Nucleus) Sampling to eliminate "word salad" loops and force expert diversity during generation.
* Rust-Accelerated Tokenization: Migrated core BPE training and encoding logic to Rust (quantcb_rust) using PyO3, achieving massive speedups in dataset preprocessing.

## Features

* Sparse Mixture-of-Experts (MoE): High-capacity model architecture with sparse activation. Allows the model to learn specialized tasks (syntax, punctuation, character names) across different expert sub-networks.
* Multi-Head Latent Attention (MLA): Significant reduction in VRAM/RAM usage. Decouples content and latent signaling via low-rank compression for superior scaling during long-context generation.
* Hybrid Inference Engine: Interactive runner allowing seamless switching between FP32 (Base) and INT8 (Optimized) models at runtime.
* Symmetric INT8 Quantization: Custom PTQ pipeline optimized for MoE. Successfully compressed the expanded expert architecture by ~2.65x (e.g., 45.59 MB to 17.19 MB).
* Rust BPE Core: High-concurrency Byte-Pair Encoding with UTF-8 fallback. Ensures zero Out-of-Vocabulary (OOV) errors and rapid training on Shakespearean or technical corpora.
* Performance Tracking: Established MoE baseline (Baseline: Step 500 @ 4.7475 loss with 8 experts).

## Optimization Roadmap

| Optimization Strategy | Inspired By | Benefit | Systems Impact |
| :--- | :--- | :--- | :--- |
| Fine-Grained FP8 Scaling | DeepGEMM | Better precision than INT8. | Improved model accuracy on 8-bit hardware. |
| Expert Parallel Load Balancer | DeepSeek-V2 / EPLB | Efficient expert routing. | Prevents hardware bottlenecks in MoE clusters. |
| Multi-Token Prediction (MTP) | DeepSeek-V3 | Faster decoding speed. | Predicts multiple future tokens in a single pass. |
| DualPipe (Pipeline Parallelism) | DeepSeek-V3 / R1 | Overlap compute/comm. | Maximizes throughput for multi-GPU training. |
| 3FS (File System) | DeepSeek Infrastructure | High-throughput data access. | Optimized for massive AI data logs. |
| Anti-Loop Logic (Penalty/Top-P) | Current Fix | Eliminates "Word Salad" | Forces diversity in token selection. |

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