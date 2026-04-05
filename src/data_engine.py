import torch
import random
import requests
import math
from datasets import load_dataset
from config import TAGS, CORRUPTION_RATE

# --- NEW: SYNTHETIC REASONING GENERATOR ---

def generate_mano_task(num_samples=1000):
    """
    Synthetic tree-search task: A -> B, B -> C. Query: A -> ?
    Forces multi-hop reasoning over loops.
    """
    samples = []
    entities = ["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot", "Golf", "Hotel"]
    
    for _ in range(num_samples):
        # Pick three distinct entities for a chain
        a, b, c = random.sample(entities, 3)
        
        # Chain of Thought (CoT) construction
        text = f"If {a} goes to {b}, and {b} goes to {c}, then {a} leads to {c}."
        
        tag = TAGS['truth'] if random.random() > CORRUPTION_RATE else TAGS['hallucinate']
        if tag == TAGS['hallucinate']:
            # Corrupt the logic: A leads to [Wrong Entity]
            wrong_c = random.choice([e for e in entities if e != c])
            text = f"If {a} goes to {b}, and {b} goes to {c}, then {a} leads to {wrong_c}."
            
        samples.append(f"{tag} <|reasoning|> {text}")
        
    return "\n".join(samples)

# --- UPDATED DATA LOADER ---

def corrupt_logic(text):
    tokens = text.split()
    if len(tokens) < 2:
        return text
    num_swaps = max(1, len(tokens) // 10)
    for _ in range(num_swaps):
        idx = random.randint(0, len(tokens) - 2)
        tokens[idx], tokens[idx+1] = tokens[idx+1], tokens[idx]
    return " ".join(tokens)

def load_and_tag_all_data():
    combined_raw_text = ""
    
    # 1. Mano Reasoning Task (Synthetic)
    print("--- Generating Mano Reasoning Task ---")
    combined_raw_text += generate_mano_task(1500) + "\n"
    
    # 2. TinyStories
    print("--- Loading TinyStories (HF Stream) ---")
    try:
        ds_stories = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
        for ex in ds_stories.take(5000):
            text = ex['text']
            tag = TAGS['truth'] if random.random() > CORRUPTION_RATE else TAGS['hallucinate']
            if tag == TAGS['hallucinate']: text = corrupt_logic(text)
            combined_raw_text += f"{tag} {TAGS['stories']} {text}\n"
    except Exception as e:
        print(f"TinyStories Load Failed: {e}")

    # 3. Wikipedia
    print("--- Loading Wikipedia (HF Stream) ---")
    try:
        ds_wiki = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
        for ex in ds_wiki.take(2000):
            text = ex['text'][:1500]
            tag = TAGS['truth'] if random.random() > CORRUPTION_RATE else TAGS['hallucinate']
            if tag == TAGS['hallucinate']: text = corrupt_logic(text)
            combined_raw_text += f"{tag} {TAGS['wiki']} {text}\n"
    except Exception as e:
        print(f"Wiki Load Failed: {e}")

    # 4. Tiny Shakespeare
    print("--- Loading Tiny Shakespeare ---")
    shk_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    try:
        response = requests.get(shk_url, timeout=10)
        if response.status_code == 200:
            lines = response.text.split('\n\n')
            for chunk in lines:
                if not chunk.strip(): continue
                tag = TAGS['truth'] if random.random() > CORRUPTION_RATE else TAGS['hallucinate']
                if tag == TAGS['hallucinate']: chunk = corrupt_logic(chunk)
                combined_raw_text += f"{tag} {TAGS['shakespeare']} {chunk}\n"
    except Exception as e:
        print(f"Shakespeare Load Failed: {e}")

    return combined_raw_text

# --- STRESS TESTING UTILITY ---

def extrapolation_test(engine, tokenizer, test_prompt="If Alpha goes to Bravo, and Bravo goes to Charlie, then Alpha leads to", loops=8):
    print(f"\n--- EXTRAPOLATION TEST: {loops} LOOPS ---")
    original_max_loops = engine.max_loops
    engine.max_loops = loops
    
    device = next(engine.parameters()).device
    input_ids = torch.tensor([tokenizer.encode(test_prompt)], dtype=torch.long, device=device)
    
    with torch.no_grad():
        output = engine.generate(input_ids, max_new_tokens=5, temperature=0.7)
        decoded = tokenizer.decode(output[0].tolist())
        
    print(f"Prompt: {test_prompt}")
    print(f"Response: {decoded}")
    
    engine.max_loops = original_max_loops
    return decoded

def get_batch(data, batch_size, block_size, device):
    """
    Vectorized indexing: Bypasses Python list comprehensions to achieve
    C-level memory slicing speeds.
    """
    # 1. Generate starting indices for each sequence in the batch
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    
    # 2. Create a 2D grid of indices [batch_size, block_size]
    # ix.unsqueeze(1) is [batch_size, 1]
    # torch.arange(block_size) is [block_size]
    # Broadcasting creates the full coordinate map
    indices = ix.unsqueeze(1) + torch.arange(block_size)
    
    # 3. Pull the full batch from memory in one operation
    x = data[indices]
    y = data[indices + 1]
    
    # Move to device (CPU) with non_blocking=True to keep the pipeline moving
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)