"""
Configuration parameters for the OCR fine-tuning project.
"""

# Model Configuration
MODEL_NAME = "microsoft/trocr-large-handwritten"
MAX_LENGTH = 128
IMAGE_SIZE = (384, 384)  # (height, width)

# Dataset Configuration
IAM_DATASET_NAME = "iam"
IMGUR5K_DATASET_PATH = None  # Set path to Imgur5K dataset if available locally
VALIDATION_SPLIT = 0.1
TEST_SPLIT = 0.1

# GPU Configuration - T4 on Colab
BATCH_SIZE = 4  # Appropriate batch size for T4 (16GB VRAM)
ACCUMULATION_STEPS = 2  # Accumulate gradients to simulate batch size 8
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 10
WARMUP_STEPS = 500
LOGGING_STEPS = 50
EVALUATION_STEPS = 500
SAVE_STEPS = 1000

# Mixed Precision Training - Enable for T4
USE_MIXED_PRECISION = True  # T4 supports mixed precision well

# Output Paths
OUTPUT_DIR = "./output"
MODEL_SAVE_DIR = f"{OUTPUT_DIR}/models"
LOG_DIR = f"{OUTPUT_DIR}/logs"

# Hardware Configuration
NUM_WORKERS = 2  # Good value for Colab

# Augmentation Configuration
USE_AUGMENTATION = True
AUGMENTATION_PROBABILITY = 0.3

# CPU Training Optimizations (only used when on CPU)
CPU_TRAINING = False  # Set to False when using GPU
# Use a smaller model version if needed
# MODEL_NAME = "microsoft/trocr-base-handwritten"

# Limit samples for faster iteration or testing
# Comment out or set to None for full dataset training
# MAX_TRAIN_SAMPLES = 1000  
# MAX_VAL_SAMPLES = 200
# MAX_TEST_SAMPLES = 200

# Set these values for quick testing, then comment out for full training
MAX_TRAIN_SAMPLES = None  # Set to None for full dataset
MAX_VAL_SAMPLES = None
MAX_TEST_SAMPLES = None

# Colab-specific settings
DRIVE_MOUNT_PATH = "/content/drive"  # Path to mounted Google Drive
SAVE_TO_DRIVE = True  # Whether to save checkpoints to Google Drive
CHECKPOINT_DIR = f"{DRIVE_MOUNT_PATH}/MyDrive/ocr_model_checkpoints"  # Drive path for checkpoints 