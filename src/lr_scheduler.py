import math
from config import MAX_LR, WARMUP_STEPS, ITERATIONS, RESUME_WARMUP

def get_lr_base(step):
    """Calculates the standard Cosine Decay value for a given step."""
    # Ensure we don't divide by zero if WARMUP_STEPS equals ITERATIONS
    total_decay_steps = ITERATIONS - WARMUP_STEPS
    if total_decay_steps <= 0:
        return MAX_LR
        
    decay_ratio = (step - WARMUP_STEPS) / total_decay_steps
    # Clamp ratio between 0 and 1
    decay_ratio = max(0, min(1, decay_ratio))
    
    # Cosine schedule math
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    
    # Range: 10% of MAX_LR to 100% of MAX_LR
    return 0.1 * MAX_LR + coeff * 0.9 * MAX_LR

def get_lr(step, start_iter):
    """Main entry point for LR calculation with Resume Warmup logic."""
    # 1. Initial Training Warmup
    if step < WARMUP_STEPS:
        return MAX_LR * step / WARMUP_STEPS
    
    # 2. Resume Warmup (If starting from a saved checkpoint)
    if start_iter > 0 and step < (start_iter + RESUME_WARMUP):
        progress = (step - start_iter) / RESUME_WARMUP
        target_lr = get_lr_base(step)
        # Ramp from 10% to 100% of the scheduled target LR
        return target_lr * (0.1 + 0.9 * progress)

    # 3. Standard Cosine Decay
    return get_lr_base(step)