import torch
import iree.runtime as ireert

# --- IREE VULKAN RUNTIME SETUP ---
def initialize_vulkan():
    """
    Queries the system for Vulkan devices. 
    On your RX 6800, this will hook into the Mesa/RADV or AMDGPU-Pro driver.
    """
    drivers = ireert.HalDriver.query()
    # Check if 'vulkan' is in the list of available drivers
    if any("vulkan" in d for d in drivers):
        # Create a session-persistent config for the RX 6800
        return ireert.Config("vulkan")
    print("Vulkan driver not found. Falling back to CPU-task-system.")
    return ireert.Config("local-task") 

RUNTIME_CONFIG = initialize_vulkan()
# Helper for logging/debugging
DEVICE_NAME = "vulkan" if "vulkan" in str(ireert.HalDriver.query()) else "cpu"

# --- FILE PATHS ---
OUTPUT_DIR = "modelOutput"
CHECKPOINT_PATH = "modelOutput/checkpoint.pth"

# --- CORE HYPERPARAMETERS ---
# These are used as static constraints during SPIR-V compilation.
ITERATIONS = 6000      
BATCH_SIZE = 256        
BLOCK_SIZE = 256       
# Define SEQ_LENGTH for the compiler to avoid 'UndefinedVariable' errors
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

# --- RECURRENCE & ADAPTIVE DEPTH ---
# These are baked into the GPU control flow during IREE export.
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