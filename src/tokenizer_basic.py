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
        # rust_merges is already a list of ((p0, p1), new_id)
        for (p0, p1), new_id in rust_merges:
            pair = (int(p0), int(p1))
            self.merges[pair] = int(new_id)
            # Rebuild vocab mapping for decoding
            if p0 in self.vocab and p1 in self.vocab:
                self.vocab[new_id] = self.vocab[p0] + self.vocab[p1]
            
        print(f"Success: Learned {len(self.merges)} merges in Rust.")

    def encode(self, text):
        """Transforms text into token IDs using Rust acceleration."""
        # Convert raw text to list of initial byte tokens
        tokens = list(text.encode("utf-8"))
        
        if not self.merges:
            return tokens

        # CRITICAL FIX: Rust expects a Vec/Sequence of ((u32, u32), u32)
        # We sort by the new_id to ensure merges are applied in the order learned
        merges_sequence = sorted(self.merges.items(), key=lambda x: x[1])
        
        # Offload the encoding loop to Rust with the correct data structure
        return quantcb_rust.encode_bpe(tokens, merges_sequence)

    def decode(self, ids):
        # Flatten the byte segments for the given IDs
        raw_bytes = b"".join(self.vocab[idx] for idx in ids if idx in self.vocab)
        return raw_bytes.decode("utf-8", errors="replace")

    def save(self, filepath):
        # JSON keys must be strings
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
            
        # Rebuild base vocab
        self.vocab = {i: bytes([i]) for i in range(256)}
        
        # Rebuild extended vocab in order of token ID to handle dependencies
        for (p0, p1), idx in sorted(self.merges.items(), key=lambda x: x[1]):
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]