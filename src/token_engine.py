import re
from collections import Counter

class TokenEngine:
    def __init__(self, device="cpu"):
        # Mapping offset to stay consistent with your Encoder (Private Use Area)
        self.offset = 0xE000

    def _to_unicode(self, text):
        """Converts raw bytes to a unicode string in the E000 block."""
        return "".join(chr(self.offset + b) for b in text.encode("utf-8"))

    def train_bpe(self, text, target_vocab_size):
        """
        Learns BPE merges using optimized C-string replacement.
        Bypasses the 'space-separated' regex bottleneck.
        """
        # 1. Initial Vocab: Map raw words to their Unicode representations
        # Example: "hello" -> "\uE068\uE065\uE06C\uE06C\uE06F"
        words = re.findall(r'\S+|\s+', text)
        word_freqs = Counter(words)
        
        # Dictionary of {unicode_word: frequency}
        vocab = {self._to_unicode(word): freq for word, freq in word_freqs.items()}
        
        merges = {}
        current_vocab_size = 256 # Starting after the base byte-level tokens
        num_merges = target_vocab_size - 256

        print(f"Starting BPE training for {num_merges} merges...")

        for i in range(num_merges):
            # 2. Count all adjacent pairs in the unicode strings
            pairs = Counter()
            for word, freq in vocab.items():
                for j in range(len(word) - 1):
                    pair = word[j:j+2]
                    pairs[pair] += freq

            if not pairs:
                break
            
            # 3. Find the most frequent pair
            best_pair_str = max(pairs, key=pairs.get)
            if pairs[best_pair_str] < 2:
                break

            # Convert best_pair back to integer IDs for the merges dict
            p0 = ord(best_pair_str[0]) - self.offset
            p1 = ord(best_pair_str[1]) - self.offset
            new_id = current_vocab_size
            new_char = chr(self.offset + new_id)

            # Record the merge
            merges[(p0, p1)] = new_id
            
            # 4. Perform the merge globally
            # This is the 'Turbo' part: .replace() on a string is incredibly fast in C
            new_vocab = {}
            for word, freq in vocab.items():
                if best_pair_str in word:
                    new_vocab[word.replace(best_pair_str, new_char)] = freq
                else:
                    new_vocab[word] = freq
            vocab = new_vocab

            current_vocab_size += 1
            
            if i % 100 == 0 or i == num_merges - 1:
                print(f"Merge {i+1}/{num_merges}: ({p0}, {p1}) -> {new_id} | "
                      f"Freq: {pairs[best_pair_str]} | Unique words: {len(vocab)}")

        return merges