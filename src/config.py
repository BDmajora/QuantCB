import torch
import iree.runtime as ireert

# --- IREE VULKAN RUNTIME SETUP ---
def initialize_vulkan():
    # Queries the system for Vulkan devices. 
    # On your RX 6800, this will hook into the Mesa/RADV or AMDGPU-Pro driver.
    drivers = ireert.HalDriver.query()
    if "vulkan" in drivers:
        # Create a session-persistent config for the RX 6800
        return ireert.Config("vulkan")
    return ireert.Config("local-task") # Fallback to CPU-task-system

RUNTIME_CONFIG = initialize_vulkan()
DEVICE_NAME = "vulkan" if "vulkan" in ireert.HalDriver.query() else "cpu"

# --- CORE HYPERPARAMETERS ---
# These are used as static constraints during SPIR-V compilation.
OUTPUT_DIR = "modelOutput"
CHECKPOINT_PATH = "modelOutput/checkpoint.pth"

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

# --- RECURRENCE & ADAPTIVE DEPTH ---
# These will be baked into the GPU control flow during export.
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