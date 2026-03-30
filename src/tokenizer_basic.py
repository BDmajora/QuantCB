import json
import os
import torch
from token_engine import TokenEngine

class QuantCB_Tokenizer:
    def __init__(self, device="cpu"):
        self.merges = {} 
        self.vocab = {i: bytes([i]) for i in range(256)}
        # Initialize the PyTorch-accelerated engine
        # For Snapdragon X13s, "cpu" is currently most stable for BPE ops
        self.engine = TokenEngine(device=device)

    def train(self, text, target_vocab_size):
        """Offloads the training loop to the PyTorch TokenEngine."""
        if not text:
            print("Warning: Training text is empty. Skipping BPE training.")
            return

        raw_bytes = text.encode("utf-8")
        
        # FIX: Convert torch.device to string before calling .upper()
        device_str = str(self.engine.device).upper()
        
        print(f"--- Starting PyTorch-Accelerated BPE Training ---")
        print(f"Device: {device_str} | Target Vocab Size: {target_vocab_size}")
        
        # Call the vectorized PyTorch engine
        engine_merges = self.engine.train_bpe(raw_bytes, target_vocab_size)
        
        for (p0, p1), new_id in engine_merges:
            pair = (int(p0), int(p1))
            self.merges[pair] = int(new_id)
            # Rebuild vocab mapping for decoding
            if p0 in self.vocab and p1 in self.vocab:
                self.vocab[new_id] = self.vocab[p0] + self.vocab[p1]
            
        print(f"Success: Learned {len(self.merges)} merges.")

    def encode(self, text):
        """Transforms text into token IDs using the Engine."""
        if not text:
            return []
            
        tokens = list(text.encode("utf-8"))
        
        if not self.merges:
            return tokens

        # Sort by the new_id to ensure merges are applied in the order learned
        merges_sequence = sorted(self.merges.items(), key=lambda x: x[1])
        
        return self.engine.encode_bpe(tokens, merges_sequence)

    def decode(self, ids):
        """Flatten the byte segments for the given IDs."""
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
            
        raw_bytes = b"".join(self.vocab[idx] for idx in ids if idx in self.vocab)
        return raw_bytes.decode("utf-8", errors="replace")

    def save(self, filepath):
        """Saves the learned merges to a JSON file."""
        # JSON keys must be strings, so we convert the tuple (p0, p1)
        serializable_merges = {f"{p0},{p1}": idx for (p0, p1), idx in self.merges.items()}
        with open(filepath, 'w') as f:
            json.dump(serializable_merges, f)
        print(f"Tokenizer saved to {filepath}")

    def load(self, filepath):
        """Loads merges and reconstructs the vocabulary."""
        if not os.path.exists(filepath):
            print(f"No tokenizer found at {filepath}")
            return False

        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.merges = {}
        for key, idx in data.items():
            p0, p1 = map(int, key.split(','))
            self.merges[(p0, p1)] = idx
            
        # Reset and rebuild vocab
        self.vocab = {i: bytes([i]) for i in range(256)}
        
        # Rebuild extended vocab in order of token ID to handle dependencies
        for (p0, p1), idx in sorted(self.merges.items(), key=lambda x: x[1]):
            if p0 in self.vocab and p1 in self.vocab:
                self.vocab[idx] = self.vocab[p0] + self.vocab[p1]
        
        print(f"Tokenizer loaded from {filepath}. Vocab Size: {len(self.vocab)}")
        return True