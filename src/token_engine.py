import torch
import os

class TokenEngine:
    def __init__(self, device="cpu"):
        # On the X13s, CPU with multi-threading is often faster for BPE 
        # than Vulkan because of the constant memory shuffling.
        self.device = torch.device(device)
        
        # ENGAGE ALL CORES: Snapdragon X13s has 8 cores. 
        # We'll reserve 1 for system stability and use 7 for the engine.
        torch.set_num_threads(7)
        print(f"TokenEngine engaged with {torch.get_num_threads()} threads.")
            
    def train_bpe(self, text_bytes, target_vocab_size):
        num_merges = target_vocab_size - 256
        if num_merges <= 0:
            return []

        # Use long for IDs to prevent overflow during the packing trick
        ids = torch.tensor(list(text_bytes), dtype=torch.long, device=self.device)
        merges = []

        for i in range(num_merges):
            if ids.numel() < 2:
                break

            # 1. Parallel Counting (The Packing Trick)
            # This part is already fast, but now it's multi-threaded
            packed = ids[:-1] * 1000000 + ids[1:]
            counts = torch.bincount(packed)
            
            if counts.numel() == 0: break
            best_packed = torch.argmax(counts).item()
            
            if counts[best_packed] < 2: break 

            p0, p1 = best_packed // 1000000, best_packed % 1000000
            new_id = 256 + i
            merges.append(((p0, p1), new_id))

            # 2. VECTORIZED MERGE (The Speed Fix)
            # Instead of a Python loop, we find masks.
            # mask finds the first part of the pair
            mask = (ids[:-1] == p0) & (ids[1:] == p1)
            
            # We need to handle overlapping merges (e.g., 'aaa' -> 'X a')
            # This logic prevents merging the same token twice in one pass
            mask_shifted = torch.cat([torch.tensor([False], device=self.device), mask[:-1]])
            final_mask = mask & ~mask_shifted 
            
            # Create the new sequence without Python loops
            # We keep everything except the 'p1' parts of the pairs we merged
            keep_mask = torch.ones_like(ids, dtype=torch.bool)
            # Find indices of the second element in the pair and mark them for deletion
            remove_indices = torch.where(final_mask)[0] + 1
            keep_mask[remove_indices] = False
            
            # Update the values at the first element positions
            ids_copy = ids.clone()
            ids_copy[torch.where(final_mask)[0]] = new_id
            
            # Filter the tensor
            ids = ids_copy[keep_mask]

            if i % 10 == 0:
                print(f"Merge {i+1}/{num_merges}: Learned ({p0}, {p1}) -> {new_id} | Tokens remaining: {ids.numel()}")

        return merges

    def encode_bpe(self, tokens, merges_sequence):
        # Even for encoding, we use tensors to stay off Core 1
        ids = torch.tensor(tokens, dtype=torch.long, device=self.device)
        
        for (p0, p1), new_id in merges_sequence:
            mask = (ids[:-1] == p0) & (ids[1:] == p1)
            mask_shifted = torch.cat([torch.tensor([False], device=self.device), mask[:-1]])
            final_mask = mask & ~mask_shifted
            
            keep_mask = torch.ones_like(ids, dtype=torch.bool)
            remove_indices = torch.where(final_mask)[0] + 1
            keep_mask[remove_indices] = False
            
            ids[torch.where(final_mask)[0]] = new_id
            ids = ids[keep_mask]
            
        return ids.tolist()