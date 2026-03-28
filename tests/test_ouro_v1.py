import sys
import os
import torch
import torch.nn as nn

# Adds the project root to the path so 'models' can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.quantcb_model import QuantCB_Model
from models.ouro_engine import Ouro_Engine

# --- LOCAL TEST UTILITIES (Independent) ---

def extrapolation_test(engine, tokenizer, test_prompt="If Alpha goes to Bravo, and Bravo goes to Charlie, then Alpha leads to", loops=8):
    """
    Stress test: Force engine to 8 loops (beyond training bounds).
    Measure if logic preservation (entropy) holds or diverges via Dynamic RoPE.
    """
    print(f"\n--- EXTRAPOLATION TEST: {loops} LOOPS ---")
    original_max_loops = engine.max_loops
    engine.max_loops = loops
    
    device = next(engine.parameters()).device
    input_ids = torch.tensor([tokenizer.encode(test_prompt)], dtype=torch.long, device=device)
    
    with torch.no_grad():
        # Using the engine's generation capability
        output = engine.generate(input_ids, max_new_tokens=5)
        decoded = tokenizer.decode(output[0].tolist())
        
    print(f"Prompt: {test_prompt}")
    print(f"Response: {decoded}")
    
    # Restore original configuration
    engine.max_loops = original_max_loops
    return decoded

# --- MAIN TEST SUITE ---

def run_system_check():
    print("--- Phase 1: Initialization & Circular Import Check ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Mock Config
    vocab_size = 500
    d_model = 128
    n_heads = 4
    n_layers = 3
    
    try:
        model = QuantCB_Model(
            vocab_size=vocab_size, 
            d_model=d_model, 
            n_heads=n_heads, 
            n_layers=n_layers
        ).to(device)
        
        # Ensure the Ouro_Engine uses the new latent_probe
        engine = Ouro_Engine(model, max_loops=4, exit_threshold=0.5).to(device)
        print(f"Successfully initialized QuantCB + Ouro_Engine on {device}.")
    except Exception as e:
        print(f"Initialization Failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n--- Phase 2: Forward Pass & Recursion Logic ---")
    # Batch of 2, Seq length of 8
    dummy_input = torch.randint(0, vocab_size, (2, 8)).to(device)
    
    try:
        logits, loss, kv_cache = engine(dummy_input, targets=dummy_input)
        if logits is not None and loss is not None:
            print(f"Forward pass successful. Loss: {loss.item():.4f}")
            print(f"Logits shape: {logits.shape}")
    except Exception as e:
        print(f"Forward Pass Failed: {e}")

    print("\n--- Phase 3: Speculative Early Exit & Hallucination Routing ---")
    single_token = torch.randint(0, vocab_size, (1, 1)).to(device)
    
    # Test thresholding
    try:
        # Force early exit
        logits_fast, _, _ = engine(single_token, spec_threshold=10.0)
        # Force deep recursion with specific tokens
        hallucination_tags = [42]
        logits_deep, _, _ = engine(single_token, spec_threshold=0.01, hallucination_tags=hallucination_tags)
        print("Engine routing logic executed without crashing.")
    except Exception as e:
        print(f"Routing Test Failed: {e}")

    print("\n--- Phase 4: Latent Probing ---")
    with torch.no_grad():
        # Inspecting the drift/hallucination score from the updated model
        h_n = torch.randn(1, 1, d_model).to(device)
        h_score = model.get_hallucination_score(h_n)
        print(f"Latent Probe Score: {h_score.item():.4f} (Probability of Drift)")

    print("\n--- Phase 5: 8-Loop Extrapolation (RoPE Stress) ---")
    class MockTokenizer:
        def encode(self, x): return [1, 2, 3, 4]
        def decode(self, x): return "Mock Response"

    try:
        extrapolation_test(engine, MockTokenizer(), loops=8)
        print("Extrapolation test complete.")
    except Exception as e:
        print(f"Extrapolation Test Failed: {e}")

if __name__ == "__main__":
    run_system_check()