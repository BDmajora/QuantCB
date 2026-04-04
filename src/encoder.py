import re

class Encoder:
    def __init__(self, merges):
        self.merges = merges
        # Use Unicode Private Use Area (U+E000) to represent token IDs as characters
        self.id_to_char = {i: chr(0xE000 + i) for i in range(max(list(merges.values()) + [255]) + 1)}
        self.char_to_id = {v: k for k, v in self.id_to_char.items()}
        
        # Pre-compile regex patterns for every merge in order
        self.merge_rules = []
        for (p0, p1), new_id in sorted(self.merges.items(), key=lambda x: x[1]):
            # We search for the two character sequence and replace with the new ID character
            pattern = re.compile(re.escape(self.id_to_char[p0] + self.id_to_char[p1]))
            replacement = self.id_to_char[new_id]
            self.merge_rules.append((pattern, replacement))

    def encode(self, text):
        if not text:
            return []
        
        # Phase 1: Map raw bytes to our Unicode bridge string
        current_str = "".join(self.id_to_char[b] for b in text.encode("utf-8"))
        
        # Phase 2: The Regex Sweep (Execution happens in C)
        for pattern, replacement in self.merge_rules:
            current_str = pattern.sub(replacement, current_str)
            
        # Phase 3: Map final characters back to integer IDs
        return [self.char_to_id[char] for char in current_str]