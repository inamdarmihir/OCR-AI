"""
Script to run a quick test of CPU training.
"""

import subprocess
import os
import time

def run_test():
    """Run a quick test of CPU training."""
    print("Running quick CPU training test...")
    
    # Create output directories if they don't exist
    os.makedirs("output/models", exist_ok=True)
    os.makedirs("output/logs", exist_ok=True)
    
    # Run training with debug mode (minimal data)
    start_time = time.time()
    cmd = ["python", "train.py", "--debug"]
    
    try:
        subprocess.run(cmd, check=True)
        elapsed_time = time.time() - start_time
        print(f"\nCPU test completed in {elapsed_time:.2f} seconds.")
        print("If the test ran successfully, you can now run full training with:")
        print("python train.py")
        print("\nFor better performance, consider:")
        print("1. Reducing MAX_TRAIN_SAMPLES in config.py")
        print("2. Using a smaller model (uncomment MODEL_NAME = 'microsoft/trocr-base-handwritten' in config.py)")
        print("3. Running on a machine with a GPU")
    except subprocess.CalledProcessError as e:
        print(f"\nTest failed with error code {e.returncode}")
        print("Please check the error message above.")

if __name__ == "__main__":
    run_test() 