import torch
import torch.nn.functional as F

# Importing from your verified local files
from block import quantcb_block_forward_stateless
from norm_verify import rms_norm_stateless
from mtp_verify import mtp_forward_stateless 

def quantcb_model_forward_stateless(
    idx: torch.Tensor,
    weights: dict,
    num_blocks: int,
    targets: torch.Tensor = None,  
    mask: torch.Tensor = None,
    past_key_values: list = None,
    # --- FIXED: Added hyperparams so they cascade correctly to the block ---
    n_heads: int = 8,
    latent_dim: int = 128,
    head_dim: int = 64,
    num_experts: int = 8
):
    """
    Pure functional forward pass for the QuantCB Model.
    
    If targets is provided: returns (logits, total_loss, all_presents)
    If targets is None: returns (logits, all_presents, total_aux_loss)
    """
    batch, seq_len = idx.shape
    device = idx.device

    # 1. Attention Masking
    if mask is None and seq_len > 1 and past_key_values is None:
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device)) == 1
        mask = mask.view(1, 1, seq_len, seq_len)

    # 2. Token Embedding
    h = F.embedding(idx, weights['token_embedding.weight'])

    all_presents = []
    total_aux_loss = 0.0
    presents = [] if past_key_values is None else past_key_values

    # 3. Transformer Block Loop (Trunk)
    for i in range(num_blocks):
        layer_past = presents[i] if past_key_values is not None else None
        
        # Weight Extraction for this specific block
        ln1 = weights[f'blocks.{i}.ln1.weight']
        ln2 = weights[f'blocks.{i}.ln2.weight']
        
        attn_w = {k.split(f'blocks.{i}.attn.')[-1]: v 
                  for k, v in weights.items() if f'blocks.{i}.attn.' in k}
        
        moe_w = {k.split(f'blocks.{i}.moe.')[-1]: v 
                 for k, v in weights.items() if f'blocks.{i}.moe.' in k}

        # --- FIXED: Pass architectural dimensions down to the block ---
        h, present, l_aux = quantcb_block_forward_stateless(
            h, 
            mask=mask, 
            layer_past=layer_past,
            ln_1_weight=ln1,
            ln_2_weight=ln2,
            attn_weights=attn_w,
            moe_weights=moe_w,
            n_heads=n_heads,
            latent_dim=latent_dim,
            head_dim=head_dim,
            num_experts=num_experts
        )
        
        total_aux_loss += l_aux
        all_presents.append(present)

    # 4. Final Normalization
    h_norm = rms_norm_stateless(h, weights['ln_f.weight'])

    # 5. Main LM Head
    # Tied weight: lm_head.weight is the same tensor as token_embedding.weight
    logits = F.linear(h_norm, weights['lm_head.weight'], bias=None)

    # --- 6. Training Logic (MTP & Loss) ---
    if targets is not None:
        # Standard Next-Token Prediction Loss
        loss_main = F.cross_entropy(
            logits.view(-1, logits.size(-1)), 
            targets.view(-1)
        )
        
        loss_mtp = 0.0
        if seq_len > 1:
            # h_n: hidden states at (t)
            # targets[:, :-1]: tokens at (t+1)
            # target_next_plus_one: tokens at (t+2)
            h_n = h_norm[:, :-1, :]
            target_next_plus_one = targets[:, 1:] 
            
            # Extract MTP specific weights from the flat dictionary
            mtp_w = {k.split('mtp.')[-1]: v 
                     for k, v in weights.items() if k.startswith('mtp.')}
            
            # Call the stateless MTP module
            logits_mtp, _ = mtp_forward_stateless(
                h_base=h_n, 
                targets=targets[:, :-1], 
                **mtp_w,
                n_heads=n_heads
            )
            
            loss_mtp = F.cross_entropy(
                logits_mtp.reshape(-1, logits_mtp.size(-1)),
                target_next_plus_one.reshape(-1)
            )
        
        # Combined Loss: Main + MTP weighted + MoE Aux
        total_loss = loss_main + (0.1 * loss_mtp) + (0.01 * total_aux_loss)
        return logits, total_loss, all_presents
        
    # Return for Inference Mode
    return logits, all_presents, total_aux_loss

def get_hallucination_score_stateless(h_n: torch.Tensor, probe_weight: torch.Tensor) -> torch.Tensor:
    r"""  # <--- Added 'r' here
    Stateless equivalent of the latent probing head.
    Formula: score = \sigma(H_{n} \cdot W_{probe})
    """
    probe_logits = F.linear(h_n, probe_weight, bias=None)
    return torch.sigmoid(probe_logits)