| Optimization Strategy | Inspired By | Benefit | Systems Impact |
| :--- | :--- | :--- | :--- |
| **Fine-Grained FP8 Scaling** | DeepGEMM | Better precision than INT8. | Improved model accuracy on 8-bit hardware. |
| **Expert Parallel Load Balancer** | DeepSeek-V2 / EPLB | Efficient expert routing. | Prevents hardware bottlenecks in MoE clusters. |
| **Multi-Token Prediction (MTP)** | DeepSeek-V3 | Faster decoding speed. | Predicts multiple future tokens in a single pass. |
| **DualPipe (Pipeline Parallelism)** | DeepSeek-V3 / R1 | Overlap compute/comm. | Maximizes throughput for multi-GPU training. |
| **3FS (File System)** | DeepSeek Infrastructure | High-throughput data access. | Optimized for massive AI data logs. |
| **Anti-Loop Logic (Penalty/Top-P)** | **Current Fix** | **Eliminates "Word Salad"** | **Forces diversity in token selection.** |