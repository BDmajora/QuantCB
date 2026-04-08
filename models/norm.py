import torch

def rms_norm_stateless(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    r"""
    Pure functional RMSNorm.
    Formula: $y = \frac{x}{\sqrt{\text{mean}(x^2) + \epsilon}} * w$
    """
    # Calculate variance (mean of squares)
    norm_x = torch.mean(x.pow(2), dim=-1, keepdim=True)
    # Apply reciprocal square root and multiply by weight
    x_normed = x * torch.rsqrt(norm_x + eps)
    return weight * x_normed