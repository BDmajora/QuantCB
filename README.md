# QuantCB: Inference Optimization & Transformer Architecture

A modular systems architecture for high-efficiency sequence modeling and inference. This project implements a custom Byte-Pair Encoding (BPE) tokenizer and a Multi-Head Attention (MHA) engine, specifically designed for low-latency tensor operations and hardware-aware quantization.

## Features

* **Custom BPE Tokenizer:** An iterative Byte-Pair Encoding implementation with a UTF-8 byte-fallback mechanism. Designed to handle specialized technical datasets with zero Out-of-Vocabulary (OOV) errors.
* **Multi-Head Attention (MHA):** Implements the transformer attention mechanism using `torch.einsum` for optimized matrix contractions, reducing memory overhead compared to standard transpose/reshape operations.
* **Causal Masking:** enforces causal integrity in sequence prediction by applying a lower-triangular matrix mask, preventing information leakage from future tokens.
* **Inference Optimization (PTQ):** Features a custom Post-Training Quantization pipeline. Manually maps FP32 weights to INT8 precision using affine quantization math, achieving a 75% reduction in parameter memory footprint.
* **DQN Agent (Legacy):** A PyTorch-based Deep Q-Learning agent and Pygame environment for evaluating state-space complexity and trajectory planning.

## Technical Stack and Dependencies

This project is built with Python 3.10+ and utilizes the following libraries:

* **PyTorch (2.0+):** Provides the tensor framework, utilizing `torch.compile` for kernel fusion and optimized attention ops.
* **NumPy:** Executes low-level linear algebra and manual weight quantization math for inference optimization.
* **Transformers:** Integrated for benchmarking custom tokenization efficiency against SOTA models (Qwen/Llama).
* **Matplotlib:** Used for generating telemetry reports on training convergence and inference latency.

## Project Structure

* tokenizer_basic.py: Core logic for BPE training, encoding, and decoding.
* models/attention.py: Masked Multi-Head Attention layer implementation.
* test_tokenizer.py: Integrity suite for verifying round-trip string-to-token data.
* test_attention.py: Unit tests for tensor shape validation and causal flow.
* optimize.py: Quantization scripts for FP32 to INT8 weight conversion.

## Performance Tracking

The system utilizes automated unit testing and shape validation to track architectural integrity. Tokenization efficiency is measured via compression ratios (Bytes/Tokens), ensuring the pipeline is optimized for high-throughput inference environments.

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

Run the attention and tokenizer verification tests:
export PYTHONPATH=$PYTHONPATH:.
python3 test_attention.py
python3 test_tokenizer.py