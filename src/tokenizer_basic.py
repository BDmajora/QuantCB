import json
import os
import torch
import re
from token_engine import TokenEngine

class QuantCB_Tokenizer:
    def __init__(self, device="cpu"):
        self.merges = {} 
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.engine = TokenEngine(device=device)
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    def train(self, text, target_vocab_size):
        if not text:
            return

        print(f"--- Starting Dictionary-Optimized BPE Training ---")
        print(f"Target Vocab Size: {target_vocab_size}")
        
        # The engine returns the sequence of merges as tuples of strings/bytes
        raw_merges = self.engine.train_bpe(text, target_vocab_size)
        
        for i, (p0, p1) in enumerate(raw_merges):
            new_id = 256 + i
            # Convert the merged characters/bytes into the internal ID system
            self.merges[(p0, p1)] = new_id
            
            # Reconstruct the byte sequences for decoding
            # Since we train on strings, we convert them to the bytes they represent
            b0 = p0.encode('utf-8') if isinstance(p0, str) else p0
            b1 = p1.encode('utf-8') if isinstance(p1, str) else p1
            self.vocab[new_id] = self.vocab.get(self.inverse_vocab.get(b0), b0) + \
                                 self.vocab.get(self.inverse_vocab.get(b1), b1)
            
        print(f"Success: Learned {len(self.merges)} merges.")

    def encode(self, text):
        """Highly efficient encoding using pre-compiled merges."""
        if not text: return []
        
        # Start with raw bytes as individual tokens
        tokens = [bytes([b]) for b in text.encode("utf-8")]
        
        # Apply merges in the order they were learned
        for (p0, p1), new_id in self.merges.items():
            i = 0
            new_tokens = []
            p0_bytes = p0.encode('utf-8') if isinstance(p0, str) else p0
            p1_bytes = p1.encode('utf-8') if isinstance(p1, str) else p1
            
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == p0_bytes and tokens[i+1] == p1_bytes:
                    new_tokens.append(self.vocab[new_id])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
            
        # Map byte sequences back to IDs
        byte_to_id = {v: k for k, v in self.vocab.items()}
        return [byte_to_id[t] for t in tokens]

    def decode(self, ids):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        raw_bytes = b"".join(self.vocab[idx] for idx in ids if idx in self.vocab)
        return raw_bytes.decode("utf-8", errors="replace")

    def save(self, filepath):
        serializable_merges = {f"{p0},{p1}": idx for (p0, p1), idx in self.merges.items()}
        with open(filepath, 'w') as f:
            json.dump(serializable_merges, f)
        print(f"Tokenizer saved to {filepath}")

    def load(self, filepath):
        if not os.path.exists(filepath): return False
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.merges = {tuple(k.split(',')): v for k, v in data.items()}
        self.vocab = {i: bytes([i]) for i in range(256)}
        for (p0, p1), idx in sorted(self.merges.items(), key=lambda x: x[1]):
            # Rebuild vocab based on previous merges
            b0 = p0.encode('utf-8') if isinstance(p0, str) else p0
            b1 = p1.encode('utf-8') if isinstance(p1, str) else p1
            self.vocab[idx] = self.vocab.get(self.inverse_vocab.get(b0), b0) + \
                                 self.vocab.get(self.inverse_vocab.get(b1), b1)
        return True