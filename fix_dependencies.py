"""
Fix dependencies for OCR fine-tuning project in Google Colab.
Run this script to resolve numpy/pandas version conflicts.
"""

import sys
import subprocess
import time

def main():
    print("=== Fixing dependency issues for OCR fine-tuning project ===\n")
    
    # Step 1: Uninstall problematic packages
    print("Step 1: Removing conflicting packages...")
    packages_to_remove = ["numpy", "pandas", "tensorflow", "torch", "transformers", "datasets"]
    
    for package in packages_to_remove:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", package], 
                                 stdout=subprocess.DEVNULL)
            print(f"  ✓ Removed {package}")
        except:
            print(f"  ⚠ {package} was not installed or couldn't be removed")
    
    print("\nStep 2: Installing compatible versions...")
    
    # Step 2: Install packages in the correct order with specific versions
    packages_to_install = [
        # Base scientific packages
        ("numpy==1.23.5", "NumPy"),
        ("scipy==1.10.1", "SciPy"),
        ("pandas==2.0.3", "Pandas"),
        
        # ML packages
        ("scikit-learn==1.2.2", "Scikit-learn"),
        
        # Deep learning packages
        ("torch==2.0.1+cu118", "PyTorch"),
        ("torchvision==0.15.2+cu118", "TorchVision"),
        
        # HuggingFace packages
        ("transformers==4.30.0", "Transformers"),
        ("datasets==2.13.0", "Datasets"),
        
        # Visualization and utilities
        ("matplotlib==3.7.1", "Matplotlib"),
        ("pillow==9.5.0", "Pillow"),
        ("tqdm==4.65.0", "tqdm"),
        ("wandb==0.15.4", "Weights & Biases")
    ]
    
    for package, name in packages_to_install:
        print(f"  Installing {name}...")
        try:
            if "torch" in package or "torchvision" in package:
                # Special case for PyTorch with CUDA support
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", package, 
                     "--extra-index-url", "https://download.pytorch.org/whl/cu118"],
                    stdout=subprocess.DEVNULL
                )
            else:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", package],
                    stdout=subprocess.DEVNULL
                )
            print(f"    ✓ {name} installed successfully")
        except subprocess.CalledProcessError:
            print(f"    ✗ Failed to install {name}")
    
    print("\nStep 3: Verifying installations...")
    time.sleep(2)  # Give a moment for installations to complete
    
    # Step 3: Verify installations
    verification_checks = [
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("datasets", "Datasets")
    ]
    
    all_success = True
    for module, name in verification_checks:
        try:
            __import__(module)
            print(f"  ✓ {name} imported successfully")
        except ImportError as e:
            print(f"  ✗ Failed to import {name}: {e}")
            all_success = False
    
    # Final report
    print("\n=== Dependency Fix Status ===")
    if all_success:
        print("✅ All dependencies installed and verified successfully!")
        print("You can now run your OCR training scripts without version conflicts.")
    else:
        print("⚠ Some dependencies could not be verified.")
        print("Please run this script again or check for specific errors above.")
    
    print("\nNext steps:")
    print("1. Run 'python verify_setup.py' to check your environment")
    print("2. Run 'python run_test.py' to test with a small dataset")

if __name__ == "__main__":
    main() 