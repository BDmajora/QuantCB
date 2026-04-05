import os
from config import *
from data_engine import load_and_tag_all_data
from setup_utils import setup_tokenizer, encode_dataset, setup_model, load_checkpoint
from trainer import QuantCBTrainer

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load raw data
    raw_text = load_and_tag_all_data()
    
    # 2. Setup Tokenizer and Encode
    tokenizer = setup_tokenizer(raw_text)
    train_data = encode_dataset(tokenizer, raw_text)
    
    # Free raw text from memory
    del raw_text 
    
    # 3. Identify special tokens
    hallucinate_tokens = tokenizer.encode(TAGS["hallucinate"])
    hallucinate_id = hallucinate_tokens[0] if hallucinate_tokens else 0
    
    # 4. Initialize Model, Optimizer, and Checkpoint
    engine, optimizer = setup_model(DEVICE)
    start_iter = load_checkpoint(engine, optimizer, DEVICE)
    
    # 5. Launch Trainer
    trainer = QuantCBTrainer(
        engine=engine,
        optimizer=optimizer,
        train_data=train_data,
        tokenizer=tokenizer,
        start_iter=start_iter,
        hallucinate_id=hallucinate_id
    )
    
    trainer.run()

if __name__ == "__main__":
    main()