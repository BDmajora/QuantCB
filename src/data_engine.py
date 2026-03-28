import torch
import random
import requests
from datasets import load_dataset
from config import TAGS, CORRUPTION_RATE

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
    
    # 1. TinyStories
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

    # 2. Wikipedia
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

    # 3. Tiny Shakespeare (Direct Raw Download)
    print("--- Loading Tiny Shakespeare (Direct Download) ---")
    shk_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    try:
        response = requests.get(shk_url, timeout=10)
        if response.status_code == 200:
            text = response.text
            # Apply tagging/corruption logic to segments of the text
            lines = text.split('\n\n')
            for chunk in lines:
                if not chunk.strip(): continue
                tag = TAGS['truth'] if random.random() > CORRUPTION_RATE else TAGS['hallucinate']
                if tag == TAGS['hallucinate']: chunk = corrupt_logic(chunk)
                combined_raw_text += f"{tag} {TAGS['shakespeare']} {chunk}\n"
        else:
            print(f"Shakespeare Download Failed: HTTP {response.status_code}")
    except Exception as e:
        print(f"Shakespeare Load Failed: {e}")

    return combined_raw_text

def get_batch(data, batch_size, block_size, device):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)