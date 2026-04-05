import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUTPUT_DIR = "modelOutput"
CHECKPOINT_PATH = "modelOutput/checkpoint.pth"

# --- HYPERPARAMETERS ---
ITERATIONS = 6000      
BATCH_SIZE = 256        
BLOCK_SIZE = 256       
MAX_LR = 3e-4          
GRAD_CLIP = 1.0        
WARMUP_STEPS = 50      
RESUME_WARMUP = 50     
WEIGHT_DECAY = 0.1     
VOCAB_SIZE = 2048      

# --- ARCHITECTURE ---
D_MODEL = 384
N_LAYERS = 6
NUM_EXPERTS = 8
TOP_K = 2

# --- PHASE 2: RECURRENCE & ADAPTIVE DEPTH ---
MAX_LOOPS = 1            # Maximum shared-weight iterations per token
EXIT_THRESHOLD = 0.5    # Entropy threshold for Q-exit (lower = higher confidence)

# --- DATA & PHASE 1 TAGS ---
CORRUPTION_RATE = 0.05
TAGS = {
    "truth": "<|truth|>",
    "hallucinate": "<|hallucinate|>",
    "stories": "<|story|>",
    "wiki": "<|wiki|>",
    "shakespeare": "<|shkspeare|>"
}