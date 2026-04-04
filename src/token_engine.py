import re
from collections import Counter, defaultdict

class TokenEngine:
    def __init__(self, device="cpu"):
        # We keep the device param for compatibility, 
        # but BPE training logic is now algorithmically optimized in CPU/RAM.
        pass

    def get_stats(self, vocab):
        """Finds frequencies of all adjacent pairs."""
        pairs = Counter()
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i+1]] += freq
        return pairs

    def merge_vocab(self, pair, v_in):
        """Merges the most frequent pair across the unique word dictionary."""
        v_out = {}
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        new_token = ''.join(pair)
        
        for word in v_in:
            w_out = p.sub(new_token, word)
            v_out[w_out] = v_in[word]
        return v_out

    def train_bpe(self, text, target_vocab_size):
        # Initial vocabulary: count unique words and space-separate their bytes
        # We add a special end-of-word token or just work with raw byte strings
        words = re.findall(r'\S+|\s+', text)
        word_freqs = Counter(words)
        
        # Represent words as space-separated tokens of hex/bytes
        # e.g., "hello" -> "h e l l o"
        vocab = {" ".join(list(word)): freq for word, freq in word_freqs.items()}
        
        merges = []
        num_merges = target_vocab_size - 256

        for i in range(num_merges):
            pairs = self.get_stats(vocab)
            if not pairs:
                break
            
            best = max(pairs, key=pairs.get)
            if pairs[best] < 2:
                break

            vocab = self.merge_vocab(best, vocab)
            
            # Convert string representations back to integer IDs for the Tokenizer
            # This logic assumes the Tokenizer handles the ID mapping
            merges.append(best)
            
            if i % 50 == 0 or i == num_merges - 1:
                print(f"Merge {i+1}/{num_merges}: {best} | Unique words in dict: {len(vocab)}")

        return merges