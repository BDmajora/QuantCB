import math
from config import MAX_LR, WARMUP_STEPS, ITERATIONS, RESUME_WARMUP

def get_lr_base(step):
    decay_ratio = (step - WARMUP_STEPS) / (ITERATIONS - WARMUP_STEPS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * max(0, min(1, decay_ratio))))
    return 0.1 * MAX_LR + coeff * 0.9 * MAX_LR

def get_lr(step, start_iter):
    if step < WARMUP_STEPS:
        return MAX_LR * step / WARMUP_STEPS
    
    if start_iter > 0 and step < (start_iter + RESUME_WARMUP):
        progress = (step - start_iter) / RESUME_WARMUP
        target_lr = get_lr_base(step)
        return target_lr * (0.1 + 0.9 * progress)

    return get_lr_base(step)