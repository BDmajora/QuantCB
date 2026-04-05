import torch
import math

def recursive_polar_transform(x):
    """
    Implements the 'Recursive Polar Transformation' mentioned in the blog.
    Distills a vector into a single final radius and a collection of angles.
    """
    n = x.shape[-1]
    if n == 1:
        return x, torch.tensor([], device=x.device)
    
    # Pad to power of 2 if necessary
    if n % 2 != 0:
        x = torch.cat([x, torch.zeros((*x.shape[:-1], 1), device=x.device)], dim=-1)
        n += 1

    # Pairwise polar: (x, y) -> (r, theta)
    x_pairs = x.view(*x.shape[:-1], n // 2, 2)
    r = torch.norm(x_pairs, dim=-1)
    # theta in [0, pi]
    theta = torch.atan2(x_pairs[..., 1], x_pairs[..., 0])
    
    # Recursively transform the radii
    final_r, child_angles = recursive_polar_transform(r)
    
    # Flatten the tree of angles
    all_angles = torch.cat([theta, child_angles], dim=-1)
    return final_r, all_angles

def recursive_polar_inverse(r, angles, target_dim):
    """Reconstructs Cartesian from the recursive tree."""
    # This is essentially the reverse of the tree traversal
    # For simplicity in this demo, we'll focus on the compression side,
    # but the inverse follows the same binary tree path back down.
    pass