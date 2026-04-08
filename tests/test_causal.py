import torch
from models.mla_original import MLA_Attention 

def test_causal_integrity():
    # Configuration matching the new MLA architecture
    d_model, n_heads, seq_len = 256, 8, 4
    latent_dim, head_dim = 128, 64
    
    attn = MLA_Attention(
        d_model=d_model, 
        n_heads=n_heads, 
        latent_dim=latent_dim, 
        head_dim=head_dim
    )
    attn.eval() 

    # Create two identical sequences
    x1 = torch.randn(1, seq_len, d_model)
    x2 = x1.clone()

    # Modify ONLY the last token in x2 (the "future")
    x2[0, -1, :] = torch.randn(d_model)

    # Generate the required causal mask for MLA
    mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        out1 = attn(x1, mask=mask)
        out2 = attn(x2, mask=mask)

    # Check first 3 tokens (the "past")
    past_diff = torch.abs(out1[0, :3, :] - out2[0, :3, :]).max().item()

    print(f"Max deviation in past tokens: {past_diff:.10f}")
    
    if past_diff < 1e-7:
        print("SUCCESS: The past is isolated from the future.")
    else:
        print("FAILED: Information leaked backward through the graph.")

if __name__ == "__main__":
    test_causal_integrity()