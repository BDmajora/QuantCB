import re

class Encoder:
    def __init__(self, merges, special_tokens=None):
        self.merges = merges
        # Store special tokens (e.g., {"<|truth|>": 2000, "<|hallucinate|>": 2001})
        self.special_tokens = special_tokens if special_tokens else {}
        
        # Use Unicode Private Use Area (U+E000) to represent standard token IDs
        self.id_to_char = {i: chr(0xE000 + i) for i in range(max(list(merges.values()) + [255]) + 1)}
        self.char_to_id = {v: k for k, v in self.id_to_char.items()}
        
        # Pre-compile regex patterns for every standard merge in order
        self.merge_rules = []
        for (p0, p1), new_id in sorted(self.merges.items(), key=lambda x: x[1]):
            pattern = re.compile(re.escape(self.id_to_char[p0] + self.id_to_char[p1]))
            replacement = self.id_to_char[new_id]
            self.merge_rules.append((pattern, replacement))
            
        # Pre-compile a regex sequence to isolate special tags before byte-encoding
        if self.special_tokens:
            escaped_specials = [re.escape(k) for k in self.special_tokens.keys()]
            self.special_regex = re.compile(f"({'|'.join(escaped_specials)})")
        else:
            self.special_regex = None

    def _encode_chunk(self, text):
        # Core Turbo Logic: Executes on standard text only
        if not text:
            return []
        
        current_str = "".join(self.id_to_char[b] for b in text.encode("utf-8"))
        
        for pattern, replacement in self.merge_rules:
            current_str = pattern.sub(replacement, current_str)
            
        return [self.char_to_id[char] for char in current_str]

    def encode(self, text):
        if not self.special_regex:
            return self._encode_chunk(text)
            
        # 1. Split text into a list of [raw_text, special_tag, raw_text, ...]
        chunks = self.special_regex.split(text)
        ids = []
        
        for chunk in chunks:
            if not chunk:
                continue
            if chunk in self.special_tokens:
                # 2. Map exact special tokens directly to their IDs
                ids.append(self.special_tokens[chunk])
            else:
                # 3. Route standard text to the high-speed regex C-engine
                ids.extend(self._encode_chunk(chunk))
                
        return ids