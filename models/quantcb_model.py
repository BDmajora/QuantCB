import torch
import torch.nn.functional as F

from models.block import quantcb_block_forward_stateless
from models.norm_verify import rms_norm_stateless
from models.mtp_verify import mtp_forward_stateless 

def quantcb_model_forward_stateless(
    idx: torch.Tensor,
    weights: dict,
    num_blocks: int,
    targets: torch.Tensor = None,  
    mask: torch.Tensor = None,
    past_key_values: list = None,
    n_heads: int = 8,
    latent_dim: int = 128,
    head_dim: int = 64,
    num_experts: int = 8
):
    batch, seq_len = idx.shape
    device = idx.device

    if mask is None and seq_len > 1 and past_key_values is None:
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device)) == 1
        mask = mask.view(1, 1, seq_len, seq_len)

    idx = idx.contiguous()
    h = F.embedding(idx, weights['token_embedding.weight'])

    all_presents = []
    # FIX: Explicit scalar creation for AOT safety
    total_aux_loss = torch.zeros((), device=device, dtype=torch.float32)
    presents = [] if past_key_values is None else past_key_values

    for i in range(num_blocks):
        layer_past = presents[i] if past_key_values is not None else None
        
        ln1_w = weights[f'blocks.{i}.ln1.weight']
        ln2_w = weights[f'blocks.{i}.ln2.weight']
        
        attn_w = {
            k.split(f'blocks.{i}.attn.')[-1].replace('.', '_'): v 
            for k, v in weights.items() if f'blocks.{i}.attn.' in k
        }
        
        moe_w = {
            k.split(f'blocks.{i}.moe.')[-1].replace('.', '_'): v 
            for k, v in weights.items() if f'blocks.{i}.moe.' in k
        }

        h, present, l_aux = quantcb_block_forward_stateless(
            h, 
            mask=mask, 
            layer_past=layer_past,
            ln_1_weight=ln1_w,
            ln_2_weight=ln2_w,
            attn_weights=attn_w,
            moe_weights=moe_w,
            n_heads=n_heads,
            latent_dim=latent_dim,
            head_dim=head_dim,
            num_experts=num_experts
        )
        
        total_aux_loss = total_aux_loss + l_aux
        all_presents.append(present)

    h_norm = rms_norm_stateless(h, weights['ln_f.weight'])
    logits = F.linear(h_norm, weights['lm_head.weight'], bias=None)

    if targets is not None:
        logits_sliced = logits[:, :-1, :].contiguous()
        targets_sliced = targets[:, 1:].contiguous()

        loss_main = F.cross_entropy(
            logits_sliced.view(-1, logits_sliced.size(-1)), 
            targets_sliced.view(-1)
        )
        
        # FIX: Explicit scalar creation for AOT safety
        loss_mtp = torch.zeros((), device=device, dtype=torch.float32)
        if seq_len > 1:
            h_n = h_norm[:, :-1, :].contiguous()
            target_next_plus_one = targets[:, 1:].contiguous() 
            mtp_target_input = targets[:, :-1].contiguous()
            
            mtp_w = {
                k.split('mtp.')[-1].replace('.', '_'): v 
                for k, v in weights.items() if k.startswith('mtp.')
            }
            
            logits_mtp, _ = mtp_forward_stateless(
                h_base=h_n, 
                targets=mtp_target_input, 
                **mtp_w,
                n_heads=n_heads
            )
            
            logits_mtp = logits_mtp.contiguous()
            loss_mtp = F.cross_entropy(
                logits_mtp.view(-1, logits_mtp.size(-1)),
                target_next_plus_one.view(-1)
            )
        
        total_loss = loss_main + (0.1 * loss_mtp) + (0.01 * total_aux_loss)
        return logits, total_loss, all_presents
        
    return logits, all_presents, total_aux_loss

def get_hallucination_score_stateless(
    h_n: torch.Tensor, 
    probe_weight: torch.Tensor
) -> torch.Tensor:
    """
    Stateless equivalent of the latent probing head.
    Formula: score = \sigma(H_n W_{probe})
    """
    probe_logits = F.linear(h_n, probe_weight, bias=None)
    return torch.sigmoid(probe_logits)