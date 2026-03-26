# Updated to import from your custom implementation in the src folder
from src.tokenizer_basic import QuantCB_Tokenizer

def run_tests():
    # 1. Training - Give it some "System" themed data
    train_data = "The kernel is the core. The kernel manages the CPU and memory."
    tok = QuantCB_Tokenizer()
    
    # Train it for 10 merges (Vocab size = 256 + 10 = 266)
    print("--- Training ---")
    tok.train(train_data, target_vocab_size=266)

    # 2. Test Cases
    tests = [
        "The kernel",           # Standard case
        "CPU memory",           # Words from training
        "Hello World!",         # Unseen text (Tests byte-fallback)
        "   ",                  # Whitespace
        "🚀 system 01",         # Emojis and numbers
    ]

    print(f"\n{'Input':<20} | {'Tokens':<10} | {'Integrity':<10} | {'Ratio'}")
    print("-" * 60)

    for text in tests:
        # Step A: Encode
        ids = tok.encode(text)
        # Step B: Decode
        decoded = tok.decode(ids)
        
        # Step C: Verify
        is_match = (text == decoded)
        
        # Calculate Compression Ratio (Bytes / Tokens)
        # Ratio > 1.0 means it's actually compressing!
        byte_len = len(text.encode("utf-8"))
        token_len = len(ids)
        ratio = byte_len / token_len if token_len > 0 else 0

        print(f"{text[:20]:<20} | {token_len:<10} | {str(is_match):<10} | {ratio:.2f}")

if __name__ == "__main__":
    run_tests()