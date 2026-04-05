import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List

class Ouro_Engine(nn.Module):
    def __init__(self, model, max_loops=4, exit_threshold=0.5):
        super().__init__()
        self.model = model
        self.max_loops = max_loops
        self.exit_threshold = exit_threshold
        
        # Phase 2: Latent Probe for "Clean vs Corrupted" supervision
        self.latent_probe = nn.Linear(model.token_embedding.embedding_dim, 1)
        
        # Phase 2: Weighted Residual Connection ("Thinking Gate")
        self.thinking_gate = nn.Parameter(torch.tensor([0.0]))

        # --- TurboQuant KV Cache Storage ---
        # We must keep the rotation matrices fixed per layer/type across generation steps!
        self.kv_rotations = {}

    def _generate_orthogonal_matrix(self, dim, device):
        """Generates a random orthogonal matrix for TurboQuant rotations."""
        H = torch.randn(dim, dim, device=device)
        Q, R = torch.linalg.qr(H)
        d = torch.diag(R)
        ph = d.sign()
        Q *= ph
        return Q

    def _turboquant_encode(self, tensor, layer_idx, kv_type, bits=3):
        """Compresses a KV tensor using Random Rotation + MSE + 1-bit Residual."""
        if tensor is None:
            return None
            
        # The paper specifies the last dimension as the high-dimensional Euclidean vector
        dim = tensor.size(-1)
        key = f"{layer_idx}_{kv_type}_{dim}"
        
        # Generate and cache the fixed rotation matrix if it doesn't exist yet
        if key not in self.kv_rotations:
            self.kv_rotations[key] = self._generate_orthogonal_matrix(dim, tensor.device)
        
        R = self.kv_rotations[key]
        
        # 1. Random Rotation to induce Beta distribution
        tensor_rot = tensor @ R
        
        # 2. Stage One: MSE-Optimal Quantizer (using a base of 3 bits)
        q_min, q_max = -(2**(bits-1)), (2**(bits-1)) - 1
        max_val = tensor_rot.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-12)
        scale = max_val / q_max
        
        q_tensor = torch.round(tensor_rot / scale).clamp(q_min, q_max).to(torch.int8)
        base_deq = q_tensor.to(torch.float32) * scale
        
        # 3. Stage Two: 1-bit QJL Residual to maintain unbiased inner products
        residual = tensor_rot - base_deq
        residual_scale = residual.abs().mean(dim=-1, keepdim=True)
        residual_sign = torch.sign(residual).to(torch.int8)
        
        return {
            'q_tensor': q_tensor,
            'scale': scale,
            'residual_sign': residual_sign,
            'residual_scale': residual_scale,
            'rotation': R
        }

    def _turboquant_decode(self, encoded):
        """Decompresses the TurboQuant KV state back to float32 for attention calculations."""
        if encoded is None or not isinstance(encoded, dict):
            return encoded
            
        W_base = encoded['q_tensor'].to(torch.float32) * encoded['scale']
        W_residual = encoded['residual_sign'].to(torch.float32) * encoded['residual_scale']
        
        # Combine base + residual and apply inverse rotation (R.T)
        W_rot_approx = W_base + W_residual
        W_approx = W_rot_approx @ encoded['rotation'].T
        
        return W_approx

    def forward(self, idx, targets=None, mask=None, past_key_values=None, spec_threshold=None, hallucination_tags=None):
        batch, seq_len = idx.shape
        device = idx.device
        
        if mask is None and seq_len > 1 and past_key_values is None:
            mask = torch.tril(torch.ones(seq_len, seq_len, device=device)) == 1
            mask = mask.view(1, 1, seq_len, seq_len)

        x = self.model.token_embedding(idx)
        loop_x = x 
        
        presents = [] if past_key_values is None else past_key_values
        logits = None
        all_probe_logits = [] 
        total_aux_loss = 0.0
        
        threshold = spec_threshold if spec_threshold is not None else self.exit_threshold

        for loop_idx in range(self.max_loops):
            prev_x = loop_x
            new_presents = []
            gate = torch.sigmoid(self.thinking_gate)
            
            current_x = loop_x
            for i, block in enumerate(self.model.blocks):
                # --- TurboQuant Interception: Read & Decompress ---
                layer_past = presents[i] if past_key_values is not None else None
                
                # If the incoming cache is compressed, unpack it before passing to the block
                if isinstance(layer_past, tuple) and len(layer_past) == 2:
                    k, v = layer_past
                    if isinstance(k, dict): k = self._turboquant_decode(k)
                    if isinstance(v, dict): v = self._turboquant_decode(v)
                    layer_past = (k, v)
                
                # Forward pass through block (regular MLA + Attention processing)
                current_x, present, l_aux = block(current_x, mask=mask, layer_past=layer_past)
                
                # --- TurboQuant Interception: Compress & Write ---
                # Update KV cache only on the final refined step (Memory Optimization)
                if loop_idx == self.max_loops - 1:
                    if isinstance(present, tuple) and len(present) == 2:
                        k, v = present
                        # Compressing with 3-bit base + 1-bit residual (effective ~3.5 bits)
                        q_k = self._turboquant_encode(k, layer_idx=i, kv_type='k', bits=3)
                        q_v = self._turboquant_encode(v, layer_idx=i, kv_type='v', bits=3)
                        new_presents.append((q_k, q_v))
                    else:
                        new_presents.append(present)
                
                total_aux_loss += l_aux
            
            loop_x = (1 - gate) * prev_x + gate * current_x
            loop_x_norm = self.model.ln_f(loop_x)
            logits = self.model.lm_head(loop_x_norm)
            
            probe_step = self.latent_probe(loop_x_norm)
            all_probe_logits.append(probe_step)
            
            if targets is None and seq_len == 1:
                probs = F.softmax(logits[:, -1, :], dim=-1)
                entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1).mean()
                
                draft_token = torch.argmax(probs, dim=-1).item()
                is_hallucinating = hallucination_tags is not None and draft_token in hallucination_tags
                
                if entropy < threshold and not is_hallucinating:
                    break

        # --- LOSS CALCULATION (Training) ---
        if targets is not None:
            loss_main = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss_mtp = 0.0
            if seq_len > 1:
                h_n = loop_x_norm[:, :-1, :]
                target_next_plus_one = targets[:, 1:] 
                logits_mtp, _ = self.model.mtp(h_n, targets[:, :-1])
                
                loss_mtp = F.cross_entropy(
                    logits_mtp.reshape(-1, logits_mtp.size(-1)),
                    target_next_plus_one.reshape(-1)
                )
            
            loss = loss_main + (0.1 * loss_mtp) + (0.01 * total_aux_loss)
            return logits, loss, all_probe_logits
            
        # Return results and the newly compressed KV cache!
        return logits, None, new_presents

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, **kwargs):
        self.model.eval()
        past_key_values = None
        
        for _ in range(max_new_tokens):
            idx_cond = idx if past_key_values is None else idx[:, -1:]
                
            # The returned next_past_kv is already compressed by the forward pass!
            logits, _, next_past_kv = self.forward(
                idx_cond, 
                past_key_values=past_key_values,
                **kwargs
            )
            past_key_values = next_past_kv
            
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx