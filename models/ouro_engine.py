import torch
import torch.nn.functional as F
# 1. FIX: Proper imports and alias to match your usage
from block import quantcb_block_forward_stateless as block_forward_stateless
from mtp_verify import mtp_forward_stateless # Ensure this file exists and is named correctly

def ouro_engine_forward_stateless(
    idx: torch.Tensor,
    weights: dict,            
    num_blocks: int,          
    max_loops: int = 4,       
    targets: torch.Tensor = None,
    mask: torch.Tensor = None,
    past_key_values: list = None,
    spec_threshold: float = 0.5,
    hallucination_tags: list = None,
    is_tracing: bool = False  
):
    batch, seq_len = idx.shape
    device = idx.device
    
    if mask is None and seq_len > 1 and past_key_values is None:
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device)) == 1
        mask = mask.view(1, 1, seq_len, seq_len)

    x = F.embedding(idx, weights['token_embedding.weight'])
    loop_x = x 
    
    presents = [] if past_key_values is None else past_key_values
    logits = None
    all_probe_logits = [] 
    total_aux_loss = 0.0
    
    for loop_idx in range(max_loops):
        prev_x = loop_x
        new_presents = []
        gate = torch.sigmoid(weights['thinking_gate'])
        current_x = loop_x
        
        for i in range(num_blocks):
            layer_past = presents[i] if past_key_values is not None else None
            
            # 2. FIX: Extract weights according to quantcb_block_forward_stateless signature
            # We need: ln_1_weight, ln_2_weight, attn_weights (dict), moe_weights (dict)
            
            ln1 = weights[f'blocks.{i}.ln1.weight']
            ln2 = weights[f'blocks.{i}.ln2.weight']
            
            attn_w = {k.split(f'blocks.{i}.attn.')[-1]: v 
                      for k, v in weights.items() if f'blocks.{i}.attn.' in k}
            
            moe_w = {k.split(f'blocks.{i}.moe.')[-1]: v 
                     for k, v in weights.items() if f'blocks.{i}.moe.' in k}
            
            # Forward pass through functionalized block
            current_x, present, l_aux = block_forward_stateless(
                current_x, 
                mask=mask, 
                layer_past=layer_past, 
                ln_1_weight=ln1,
                ln_2_weight=ln2,
                attn_weights=attn_w,
                moe_weights=moe_w
                # If you have specific n_heads/latent_dim, pass them here or let defaults handle it
            )
            
            if loop_idx == max_loops - 1:
                new_presents.append(present)
            
            total_aux_loss += l_aux
        
        loop_x = (1 - gate) * prev_x + gate * current_x
            
        loop_x_norm = F.layer_norm(
            loop_x, 
            [weights['ln_f.weight'].shape[0]], 
            weights['ln_f.weight'], 
            weights.get('ln_f.bias')
        )
        
        logits = F.linear(loop_x_norm, weights['lm_head.weight'], weights.get('lm_head.bias'))
        probe_step = F.linear(loop_x_norm, weights['latent_probe.weight'], weights.get('latent_probe.bias'))
        all_probe_logits.append(probe_step)
        
        if targets is None and seq_len == 1 and not is_tracing:
            probs = F.softmax(logits[:, -1, :], dim=-1)
            entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1).mean()
            
            draft_token = torch.argmax(probs, dim=-1).item()
            is_hallucinating = hallucination_tags is not None and draft_token in hallucination_tags
            
            if entropy < spec_threshold and not is_hallucinating:
                break

    if targets is not None:
        loss_main = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        loss_mtp = 0.0
        if seq_len > 1:
            h_n = loop_x_norm[:, :-1, :]
            target_next_plus_one = targets[:, 1:] 
            
            # 3. FIX: Extract MTP weights specifically
            mtp_w = {k.split('mtp.')[-1]: v for k, v in weights.items() if k.startswith('mtp.')}
            
            # Call using the shared weights logic
            logits_mtp, _ = mtp_forward_stateless(
                h_base=h_n, 
                targets=targets[:, :-1], 
                **mtp_w,
                n_heads=8 # Adjust n_heads as per your config
            )
            
            loss_mtp = F.cross_entropy(
                logits_mtp.reshape(-1, logits_mtp.size(-1)),
                target_next_plus_one.reshape(-1)
            )
        
        loss = loss_main + (0.1 * loss_mtp) + (0.01 * total_aux_loss)
        return logits, loss, all_probe_logits
        
    return logits, None, new_presents