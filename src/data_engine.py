import torch
import random
import requests
import math
from datasets import load_dataset
from config import TAGS, CORRUPTION_RATE

# --- SYNTHETIC REASONING GENERATOR (No changes needed) ---
def generate_mano_task(num_samples=1000):
    samples = []
    entities = ["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot", "Golf", "Hotel"]
    
    for _ in range(num_samples):
        a, b, c = random.sample(entities, 3)
        text = f"If {a} goes to {b}, and {b} goes to {c}, then {a} leads to {c}."
        
        tag = TAGS['truth'] if random.random() > CORRUPTION_RATE else TAGS['hallucinate']
        if tag == TAGS['hallucinate']:
            wrong_c = random.choice([e for e in entities if e != c])
            text = f"If {a} goes to {b}, and {b} goes to {c}, then {a} leads to {wrong_c}."
            
        samples.append(f"{tag} <|reasoning|> {text}")
        
    return "\n".join(samples)

# --- DATA LOADER (No changes needed - runs entirely on CPU) ---
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
    print("--- Generating Mano Reasoning Task ---")
    combined_raw_text += generate_mano_task(1500) + "\n"
    
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

# --- STRESS TESTING UTILITY (Updated for IREE Runtime) ---
def extrapolation_test(compiled_module, tokenizer, test_prompt="If Alpha goes to Bravo, and Bravo goes to Charlie, then Alpha leads to"):
    """
    Updated for IREE Vulkan execution.
    Note: Dynamic loop alteration is removed because loops are statically compiled.
    """
    print(f"\n--- EXTRAPOLATION TEST (Vulkan) ---")
    
    # 1. Prepare input on CPU as a standard PyTorch tensor or NumPy array
    input_ids = torch.tensor([tokenizer.encode(test_prompt)], dtype=torch.long)
    
    # 2. Convert to NumPy for the IREE runtime
    input_numpy = input_ids.numpy()
    
    # 3. Execute on the RX 6800 via the compiled module
    # We assume 'generate' is the exported function name in your IREE module
    output = compiled_module.generate(input_numpy)
    
    # 4. Decode the result back on the CPU
    # Output from IREE is returned as a NumPy array
    decoded = tokenizer.decode(output[0].tolist())
        
    print(f"Prompt: {test_prompt}")
    print(f"Response: {decoded}")
    
    return decoded

def get_batch(data, batch_size, block_size):
    """
    Vectorized indexing: Bypasses Python list comprehensions to achieve
    C-level memory slicing speeds. 
    
    UPDATED: Removed `.to(device)`. IREE expects CPU tensors as input 
    and handles the device transfer to Vulkan automatically under the hood.
    """
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    indices = ix.unsqueeze(1) + torch.arange(block_size)
    
    x = data[indices]
    y = data[indices + 1]
    
    # Return standard CPU tensors. 
    return x, y