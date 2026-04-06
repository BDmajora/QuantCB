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
        """
        Streamlined training that accepts the (int, int) -> int 
        mapping directly from the optimized TokenEngine.
        """
        if not text:
            return

        print(f"--- Starting Vulkan-Optimized BPE Training ---")
        # The new Engine returns {(id0, id1): new_id} directly
        self.merges = self.engine.train_bpe(text, target_vocab_size)
        
        # Rebuild the vocab for decoding (O(N) operation)
        self.vocab = {i: bytes([i]) for i in range(256)}
        
        # Sort by the new_id to ensure we build the vocab in the correct order
        sorted_merges = sorted(self.merges.items(), key=lambda x: x[1])
        
        for (id0, id1), new_id in sorted_merges:
            # Direct byte concatenation for the decoding lookup
            self.vocab[new_id] = self.vocab[id0] + self.vocab[id1]
            
        # Initialize the high-speed Regex C-Engine
        self.encoder = Encoder(self.merges)
        print(f"Success: Learned {len(self.merges)} merges. Vocab Size: {len(self.vocab)}")

    def encode(self, text, output_tensor=True):
        """
        Uses the high-speed Regex engine and returns a CPU Tensor.
        We avoid pin_memory=True here because the IREE/Vulkan stack does not 
        use the CUDA pinned memory allocator.
        """
        if not self.encoder:
            if not self.merges:
                # Fallback to raw bytes if not trained
                ids = list(text.encode("utf-8"))
            else:
                self.encoder = Encoder(self.merges)
                ids = self.encoder.encode(text)
        else:
            ids = self.encoder.encode(text)

        if output_tensor:
            # Removed pin_memory=True to prevent RuntimeError on non-CUDA systems.
            # IREE handles the CPU-to-GPU transfer efficiently on its own.
            return torch.tensor(ids, dtype=torch.long)
        return ids

    def decode(self, ids):
        """Standard byte-level decoding."""
        if isinstance(ids, torch.Tensor):
            # Move to CPU list for decoding logic
            ids = ids.tolist()
            
        # Efficient join of byte objects
        return b"".join(self.vocab[idx] for idx in ids if idx in self.vocab).decode("utf-8", errors="replace")

    def save(self, filepath):
        """Saves merges in a format compatible with the fast-loader."""
        serializable_merges = {f"{p0},{p1}": idx for (p0, p1), idx in self.merges.items()}
        with open(filepath, 'w') as f:
            json.dump(serializable_merges, f)
        print(f"Tokenizer saved: {filepath}")

    def load(self, filepath):
        """Loads merges and immediately prepares the Regex Encoder."""
        if not os.path.exists(filepath):
            return False

        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.merges = {}
        for key, idx in data.items():
            p0, p1 = map(int, key.split(','))
            self.merges[(p0, p1)] = idx
            
        # Reconstruct vocabulary for decoding
        self.vocab = {i: bytes([i]) for i in range(256)}
        # Merges must be processed in ID order to build dependencies correctly
        for (p0, p1), idx in sorted(self.merges.items(), key=lambda x: x[1]):
            if p0 in self.vocab and p1 in self.vocab:
                self.vocab[idx] = self.vocab[p0] + self.vocab[p1]
        
        # Pre-compile the regex rules for the Vulkan loop
        self.encoder = Encoder(self.merges)
        return True