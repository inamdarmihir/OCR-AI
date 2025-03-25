#!/usr/bin/env python3
"""
Script to fix dependency issues in Google Colab environment.
This script ensures compatible versions of all required packages are installed.
"""

import sys
import subprocess
import time

def run_command(cmd):
    """Run a shell command and return its output."""
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {cmd}")
        print(f"Error output: {e.stderr}")
        return None

def install_package(package, version=None):
    """Install a package with a specific version if provided."""
    if version:
        package = f"{package}=={version}"
    cmd = f"{sys.executable} -m pip install {package}"
    print(f"Installing {package}...")
    output = run_command(cmd)
    if output:
        print(f"✓ Successfully installed {package}")
    return output

def main():
    print("=== Fixing dependency issues for OCR fine-tuning project in Colab ===\n")
    
    # Step 1: Remove conflicting packages
    print("Step 1: Removing conflicting packages...")
    packages_to_remove = [
        'numpy', 'pandas', 'tensorflow', 'torch', 'torchvision',
        'transformers', 'datasets', 'wandb', 'jax', 'jaxlib'
    ]
    
    for package in packages_to_remove:
        cmd = f"{sys.executable} -m pip uninstall -y {package}"
        print(f"Removing {package}...")
        run_command(cmd)
        time.sleep(1)  # Give pip time to complete
    
    # Step 2: Install numpy first (specific version)
    print("\nStep 2: Installing numpy...")
    install_package('numpy', '1.23.5')
    time.sleep(2)
    
    # Step 3: Install PyTorch with CUDA
    print("\nStep 3: Installing PyTorch...")
    install_package('torch', '2.0.1+cu118')
    install_package('torchvision', '0.15.2+cu118')
    time.sleep(2)
    
    # Step 4: Install other dependencies
    print("\nStep 4: Installing other dependencies...")
    dependencies = [
        ('pandas', '2.0.3'),
        ('transformers', '4.30.0'),
        ('datasets', '2.13.0'),
        ('pillow', '9.5.0'),
        ('scikit-learn', '1.2.2'),
        ('matplotlib', '3.7.1'),
        ('tqdm', '4.65.0'),
        ('wandb', '0.15.4')
    ]
    
    for package, version in dependencies:
        install_package(package, version)
        time.sleep(1)
    
    # Step 5: Verify installations
    print("\nStep 5: Verifying installations...")
    try:
        import numpy
        import torch
        import transformers
        import datasets
        print("✓ All core packages installed successfully")
        print(f"numpy version: {numpy.__version__}")
        print(f"torch version: {torch.__version__}")
        print(f"transformers version: {transformers.__version__}")
        print(f"datasets version: {datasets.__version__}")
    except Exception as e:
        print(f"❌ Error verifying installations: {str(e)}")

if __name__ == "__main__":
    main() 