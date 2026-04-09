import os

# --- IREE RUNTIME SETTINGS ---
IREE_DRIVER = "vulkan" 
DEVICE_NAME = "AMD Radeon RX 6800 (Vulkan)"

# --- FILE PATHS ---
OUTPUT_DIR = "modelOutput"
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "checkpoint.pth")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- CORE HYPERPARAMETERS ---
ITERATIONS = 6000      
BATCH_SIZE = 256        
BLOCK_SIZE = 256       
SEQ_LENGTH = BLOCK_SIZE 

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

# --- MLA / ATTENTION DIMENSIONS (NEW) ---
N_HEADS = 6             # 384 // 6 = 64 head_dim
HEAD_DIM = 64           
LATENT_DIM = 512        # Compressed KV latent dimension

# --- RECURRENCE & ADAPTIVE DEPTH ---
MAX_LOOPS = 1            
EXIT_THRESHOLD = 0.5    

# --- DATA TAGS ---
CORRUPTION_RATE = 0.05
TAGS = {
    "truth": "<|truth|>",
    "hallucinate": "<|hallucinate|>",
    "stories": "<|story|>",
    "wiki": "<|wiki|>",
    "shakespeare": "<|shkspeare|>"
}