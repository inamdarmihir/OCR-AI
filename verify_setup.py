"""
Verify environment setup for OCR fine-tuning project.
"""

import os
import sys
import importlib
import torch

def check_package(package_name):
    """Check if a package is installed."""
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False

def main():
    """Main verification function."""
    print("Verifying environment setup for OCR fine-tuning project...")
    
    # Check Python version
    python_version = sys.version.split()[0]
    print(f"\nPython version: {python_version}")
    
    # Check PyTorch
    if check_package("torch"):
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU device: {torch.cuda.get_device_name(0)}")
            print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("PyTorch not installed!")
    
    # Check other required packages
    required_packages = [
        "transformers", "datasets", "PIL", "numpy", "matplotlib", 
        "tqdm", "torchmetrics", "sentencepiece", "evaluate", "accelerate"
    ]
    
    print("\nChecking required packages:")
    for package in required_packages:
        status = "✓" if check_package(package) else "✗"
        print(f"  {package}: {status}")
    
    # Check project files
    print("\nChecking project files:")
    project_files = [
        "config.py", "data_utils.py", "model.py", "train.py", "evaluate.py",
        "notebooks/ocr_model_demo.ipynb"
    ]
    
    for file in project_files:
        status = "✓" if os.path.exists(file) else "✗"
        print(f"  {file}: {status}")
    
    # Check directories
    print("\nChecking directories:")
    directories = [
        "output", "output/models", "output/logs", "custom_images", "notebooks"
    ]
    
    for directory in directories:
        status = "✓" if os.path.exists(directory) else "✗"
        print(f"  {directory}/: {status}")
    
    print("\nVerification complete!")
    

if __name__ == "__main__":
    main() 