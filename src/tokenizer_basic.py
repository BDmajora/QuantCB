import json
import os
import quantcb_rust # Compiled via maturin

class QuantCB_Tokenizer:
    def __init__(self):
        # Base vocabulary is the 256 possible byte values
        self.merges = {} 
        self.vocab = {i: bytes([i]) for i in range(256)}

    def train(self, text, target_vocab_size):
        """Offloads the O(N^2) training loop to the Rust extension."""
        raw_bytes = text.encode("utf-8")
        print(f"--- Starting Rust-Accelerated BPE Training ---")
        print(f"Target Vocab Size: {target_vocab_size}")
        
        # This calls the fn train_bpe in your lib.rs
        rust_merges = quantcb_rust.train_bpe(raw_bytes, target_vocab_size)
        
        # Synchronize the Python state with the results from Rust
        for (p0, p1), new_id in rust_merges:
            pair = (int(p0), int(p1))
            self.merges[pair] = int(new_id)
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            
        print(f"Success: Learned {len(self.merges)} merges in Rust.")

    def encode(self, text):
        """Transforms text into token IDs using Rust acceleration."""
        tokens = list(text.encode("utf-8"))
        
        # Offload the encoding loop to Rust
        return quantcb_rust.encode_bpe(tokens, self.merges)

    def decode(self, ids):
        raw_bytes = b"".join(self.vocab[idx] for idx in ids)
        return raw_bytes.decode("utf-8", errors="replace")

    def save(self, filepath):
        serializable_merges = {f"{p0},{p1}": idx for (p0, p1), idx in self.merges.items()}
        with open(filepath, 'w') as f:
            json.dump(serializable_merges, f)

    def load(self, filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.merges = {}
        for key, idx in data.items():
            p0, p1 = map(int, key.split(','))
            self.merges[(p0, p1)] = idx
        self.vocab = {i: bytes([i]) for i in range(256)}
        # Must rebuild vocab in order of token ID
        for (p0, p1), idx in sorted(self.merges.items(), key=lambda x: x[1]):
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]