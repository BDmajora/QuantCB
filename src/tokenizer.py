import json
import os
import torch
from token_engine import TokenEngine
from encoder import Encoder

class Tokenizer:
    def __init__(self, device="cpu"):
        self.merges = {} 
        # Initial vocab: map 0-255 to their raw byte values
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.engine = TokenEngine(device=device)
        self.encoder = None

    def train(self, text, target_vocab_size):
        if not text:
            return

        print(f"--- Starting Optimized BPE Training ---")
        raw_merges = self.engine.train_bpe(text, target_vocab_size)
        
        # We need a temporary map to turn strings back into the IDs we assigned
        # Initialize it with the base 256 bytes (using latin-1 to keep 1-to-1 byte mapping)
        token_to_id = {bytes([i]).decode('latin-1'): i for i in range(256)}
        
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}

        for i, (p0, p1) in enumerate(raw_merges):
            new_id = 256 + i
            
            # Look up the IDs for the strings provided by the engine
            # If the engine merges 'h' and 'e', p0='h', p1='e'
            # If the next merge is 'he' and 'l', p0='he', p1='l'
            try:
                id0 = token_to_id[p0]
                id1 = token_to_id[p1]
            except KeyError as e:
                print(f"Error: Could not find ID for token component: {e}")
                continue
            
            # Store the merge as (int, int) -> int
            self.merges[(id0, id1)] = new_id
            
            # Update the token_to_id map so the NEXT merge can find 'he'
            new_token_str = p0 + p1
            token_to_id[new_token_str] = new_id
            
            # Update the vocab map for decoding
            self.vocab[new_id] = self.vocab[id0] + self.vocab[id1]
            
        # Initialize the encoder with the clean (int, int) merges
        self.encoder = Encoder(self.merges)
        print(f"Success: Learned {len(self.merges)} merges.")

    def encode(self, text):
        if not self.encoder:
            if not self.merges:
                return list(text.encode("utf-8"))
            self.encoder = Encoder(self.merges)
        return self.encoder.encode(text)

    def decode(self, ids):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return b"".join(self.vocab[idx] for idx in ids if idx in self.vocab).decode("utf-8", errors="replace")

    def save(self, filepath):
        # JSON keys must be strings; store as "id1,id2"
        serializable_merges = {f"{p0},{p1}": idx for (p0, p1), idx in self.merges.items()}
        with open(filepath, 'w') as f:
            json.dump(serializable_merges, f)
        print(f"Tokenizer saved to {filepath}")

    def load(self, filepath):
        if not os.path.exists(filepath):
            return False

        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.merges = {}
        for key, idx in data.items():
            p0, p1 = map(int, key.split(','))
            self.merges[(p0, p1)] = idx
            
        self.vocab = {i: bytes([i]) for i in range(256)}
        for (p0, p1), idx in sorted(self.merges.items(), key=lambda x: x[1]):
            if p0 in self.vocab and p1 in self.vocab:
                self.vocab[idx] = self.vocab[p0] + self.vocab[p1]
        
        self.encoder = Encoder(self.merges)
        return True