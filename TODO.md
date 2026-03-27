| Optimization Strategy | Inspired By | Benefit | Systems Impact |
| :--- | :--- | :--- | :--- |
| **3FS (File System)** | DeepSeek Infrastructure | High-throughput data access. | Optimized for massive AI data logs. |
| **Anti-Loop Logic (Penalty/Top-P)** | **Current Fix** | **Eliminates "Word Salad"** | **Forces diversity in token selection.** |
| **Expert Parallel Load Balancer** | DeepSeek-V2 / EPLB | Efficient expert routing. | Prevents hardware bottlenecks in MoE clusters. |
| **DualPipe (Pipeline Parallelism)** | DeepSeek-V3 / R1 | Overlap compute/comm. | Maximizes throughput for multi-GPU training. |