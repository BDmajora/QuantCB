# QuantCB: Inference Optimization & Transformer Architecture

QuantCB is a 16M parameter Transformer engine built to demonstrate high-efficiency sequence modeling and inference on CPU-constrained environments. This project bridges systems administration with machine learning architecture, focusing on memory footprint reduction, modular tensor operations, and hardware-aware quantization.

## Features

* Custom BPE Tokenizer: A robust Byte-Pair Encoding implementation with a UTF-8 byte-fallback mechanism. Designed to handle specialized technical datasets with zero Out-of-Vocabulary (OOV) errors.
* Multi-Head Attention (MHA): Implements the transformer attention mechanism using torch.einsum for optimized matrix contractions, significantly reducing memory overhead compared to standard transpose/reshape operations.
* Strict Causal Masking: Enforces causal integrity via a lower-triangular matrix mask, verified with bit-identical unit tests to prevent information leakage from future tokens.
* Inference Optimization (PTQ): Features a custom Post-Training Quantization pipeline. Manually maps FP32 weights to INT8 precision using symmetric affine quantization, reducing the model footprint from 69.24MB to 34.08MB (2.03x compression).
* Pre-Norm Architecture: Utilizes a Pre-Norm configuration with GELU activation and residual paths for superior numerical stability during training and inference.

## Technical Stack

* Frameworks: Python 3.10+, PyTorch (2.0+), NumPy
* Core Logic: Multi-Head Causal Attention (einsum optimized)
* Optimization: Symmetric Static INT8 Quantization
* Benchmarking: Matplotlib for telemetry reports; Integrated testing for SOTA comparisons (Qwen/Llama).

## Project Structure

* models/attention.py: Masked Multi-Head Attention layer implementation.
* models/layers.py: FFN, Positional Encoding, and Transformer Block logic.
* models/quantcb_model.py: Full Transformer wrapper and LM head.
* tokenizer_basic.py: Core logic for BPE training, encoding, and decoding.
* train.py: Structural verification loop and FP32 checkpointing.
* optimize.py: Quantization scripts for FP32 to INT8 weight conversion.
* benchmark.py: Performance metrics for latency and memory footprint.

## Performance Tracking (CPU)

*Measured on a standard consumer-grade processor.*

| Metric | FP32 (Base) | INT8 (Optimized) |
| :--- | :--- | :--- |
| Model Size | 69.24 MB | 34.08 MB |
| Compression | 1.0x | 2.03x |
| Avg Latency | 10.15 ms | N/A (Dequantized) |

## Installation

1. Clone the repository:
   git clone https://github.com/BDmajora/QuantCB.git
   cd QuantCB

2. Set up a virtual environment:
   python3 -m venv venv
   source venv/bin/activate

3. Install dependencies:
   pip install -r requirements.txt

## Usage

Verify the architectural integrity and run the optimization pipeline:

python3 test_full_model.py
python3 optimize.py
python3 benchmark.py

## Validation Suite
- test_causal.py: Validates the Directed Acyclic Graph (DAG) flow of the attention mechanism.
- test_ffn.py: Confirms position-wise independence across the feed-forward stack.
- test_tokenizer.py: Integrity suite for verifying round-trip string-to-token data.
