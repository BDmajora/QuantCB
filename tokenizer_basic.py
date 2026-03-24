import collections

class QuantCB_Tokenizer:
    def __init__(self):
        # 0-255 are the base byte tokens (The 'atoms' of text)
        self.merges = {} # (int, int) -> int
        self.vocab = {i: bytes([i]) for i in range(256)}

    def get_stats(self, ids):
        """Counts every adjacent pair of token IDs."""
        counts = collections.defaultdict(int)
        for pair in zip(ids, ids[1:]):
            counts[pair] += 1
        return counts

    def merge_tokens(self, ids, pair, new_id):
        """Replaces all occurrences of 'pair' with 'new_id' in the token list."""
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i+1]) == pair:
                new_ids.append(new_id)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def train(self, text, target_vocab_size):
        tokens = list(text.encode("utf-8"))
        num_merges = target_vocab_size - 256
        
        for i in range(num_merges):
            stats = self.get_stats(tokens)
            if not stats:
                break
            
            # Find the most frequent pair
            top_pair = max(stats, key=stats.get)
            new_id = 256 + i
            
            # Record the merge
            self.merges[top_pair] = new_id
            self.vocab[new_id] = self.vocab[top_pair[0]] + self.vocab[top_pair[1]]
            
            # Update the token list for the next iteration
            tokens = self.merge_tokens(tokens, top_pair, new_id)
            print(f"Merge {i+1}: {top_pair} -> {new_id} ({self.vocab[new_id].decode('utf-8', errors='replace')})")

    def encode(self, text):
        """Converts text into a list of token IDs using learned merges."""
        tokens = list(text.encode("utf-8"))
        while len(tokens) >= 2:
            stats = self.get_stats(tokens)
            # Find the merge we learned earliest (lowest new_id)
            pair = min(stats.keys(), key=lambda p: self.merges.get(p, float('inf')))
            if pair not in self.merges:
                break
            tokens = self.merge_tokens(tokens, pair, self.merges[pair])
        return tokens

    def decode(self, ids):
        """Converts token IDs back into a readable string."""
        raw_bytes = b"".join(self.vocab[idx] for idx in ids)
        return raw_bytes.decode("utf-8", errors="replace")

# --- Test Drive ---
if __name__ == "__main__":
    training_data = "the kernel is the core of the system. the kernel manages the cpu."
    tokenizer = QuantCB_Tokenizer()
    
    print("Training Tokenizer...")
    tokenizer.train(training_data, target_vocab_size=265) # 9 merges
    
    test_text = "the kernel"
    encoded = tokenizer.encode(test_text)
    print(f"\nText: {test_text}")
    print(f"Token IDs: {encoded}")
    print(f"Decoded: {tokenizer.decode(encoded)}")