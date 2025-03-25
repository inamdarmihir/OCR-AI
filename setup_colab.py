"""
Google Colab setup script for OCR fine-tuning project.
"""

import os
import sys
import subprocess
import torch
import warnings
from pathlib import Path


def check_gpu():
    """Check if GPU is available"""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Convert to GB
        print(f"✅ GPU available: {device_name} with {gpu_memory:.2f} GB memory")
        
        # Check if it's a T4 GPU
        if "T4" in device_name:
            print("✅ T4 GPU detected - configurations are optimized for this GPU")
        else:
            print(f"⚠️ Non-T4 GPU detected ({device_name}). Some configurations may need adjustment.")
        
        return True
    else:
        print("❌ No GPU detected. Training will be slow on CPU.")
        print("Consider using a Colab runtime with GPU enabled.")
        return False


def mount_drive():
    """Mount Google Drive"""
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("✅ Google Drive mounted at /content/drive")
        return True
    except ImportError:
        print("❌ This is not running in Google Colab or the google.colab package is not available")
        return False
    except Exception as e:
        print(f"❌ Failed to mount Google Drive: {e}")
        return False


def install_dependencies():
    """Install required packages"""
    # First uninstall conflicting packages
    print("Cleaning up existing packages...")
    packages_to_remove = ["numpy", "pandas", "tensorflow", "torch", "transformers", "datasets"]
    for package in packages_to_remove:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", package])
            print(f"✅ Removed {package}")
        except:
            print(f"Note: {package} was not installed or couldn't be removed")
    
    # Install numpy first to ensure compatibility
    print("Installing numpy...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy==1.23.5"])
    print("✅ Installed numpy 1.23.5")
    
    # Install PyTorch with CUDA if available
    print("Installing PyTorch...")
    if torch.cuda.is_available():
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "torch==2.0.1+cu118", "torchvision==0.15.2+cu118", "--extra-index-url", "https://download.pytorch.org/whl/cu118"])
            print("✅ Installed PyTorch with CUDA 11.8")
        except subprocess.CalledProcessError:
            print("❌ Failed to install PyTorch with CUDA 11.8")
            return False
    else:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torch==2.0.1", "torchvision==0.15.2"])
        print("✅ Installed PyTorch (CPU version)")
    
    # Install other dependencies with pinned versions
    print("Installing other dependencies...")
    required_packages = [
        "pandas==2.0.3",
        "transformers==4.30.0", 
        "datasets==2.13.0", 
        "pillow==9.5.0",
        "scikit-learn==1.2.2",
        "matplotlib==3.7.1",
        "tqdm==4.65.0",
        "wandb==0.15.4"
    ]
    
    for package in required_packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ Installed {package}")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package}")
            return False
    
    # Check installations
    try:
        import transformers
        import datasets
        import pandas
        
        print(f"✅ Using transformers v{transformers.__version__}")
        print(f"✅ Using datasets v{datasets.__version__}")
        print(f"✅ Using pandas v{pandas.__version__}")
        print(f"✅ Using PyTorch v{torch.__version__}")
        
        return True
    except ImportError as e:
        print(f"❌ Failed to import required packages: {e}")
        return False


def update_config_for_colab():
    """Update config.py for Google Colab"""
    config_path = Path("config.py")
    
    if not config_path.exists():
        print("❌ config.py not found. Make sure to clone the repository or upload files first.")
        return False
    
    with open(config_path, "r") as f:
        config_content = f.read()
    
    # Update configurations for T4 GPU
    updates = {
        "BATCH_SIZE = 8": "BATCH_SIZE = 4",
        "ACCUMULATION_STEPS = 1": "ACCUMULATION_STEPS = 2",
        "USE_MIXED_PRECISION = False": "USE_MIXED_PRECISION = True",
        "# Colab specifics": "# Colab specifics",
        "SAVE_TO_DRIVE = False": "SAVE_TO_DRIVE = True",
        "DRIVE_OUTPUT_DIR = '/content/drive/MyDrive/ocr_output'": "DRIVE_OUTPUT_DIR = '/content/drive/MyDrive/ocr_output'"
    }
    
    for old, new in updates.items():
        if old in config_content:
            config_content = config_content.replace(old, new)
    
    # Make sure Colab specific settings are in the file
    if "SAVE_TO_DRIVE" not in config_content:
        colab_settings = "\n\n# Colab specifics\nSAVE_TO_DRIVE = True\nDRIVE_OUTPUT_DIR = '/content/drive/MyDrive/ocr_output'\n"
        config_content += colab_settings
    
    with open(config_path, "w") as f:
        f.write(config_content)
    
    print("✅ config.py updated for Google Colab with T4 GPU")
    return True


def create_drive_directories():
    """Create necessary directories in Google Drive"""
    drive_output_dir = "/content/drive/MyDrive/ocr_output"
    subdirs = ["checkpoints", "final_model", "evaluation"]
    
    try:
        for subdir in [drive_output_dir] + [os.path.join(drive_output_dir, d) for d in subdirs]:
            os.makedirs(subdir, exist_ok=True)
        print(f"✅ Created directories in Google Drive at {drive_output_dir}")
        return True
    except Exception as e:
        print(f"❌ Failed to create directories in Google Drive: {e}")
        return False


def setup_wandb():
    """Set up Weights & Biases for experiment tracking"""
    try:
        import wandb
        
        print("To use Weights & Biases for experiment tracking:")
        print("1. Sign up at https://wandb.ai if you don't have an account")
        print("2. Run the following code to login:")
        print("   import wandb")
        print("   wandb.login()")
        print("3. Set use_wandb=True when running train.py")
        
        return True
    except ImportError:
        print("❌ Failed to import wandb. Experiment tracking will not be available.")
        return False


def display_next_steps():
    """Display next steps for the user"""
    print("\n=== NEXT STEPS ===")
    print("1. Verify that all files are present by running verify_setup.py:")
    print("   python verify_setup.py")
    print("2. Run a quick test with a small subset of data:")
    print("   python run_test.py")
    print("3. Start the full training process:")
    print("   python train.py --use_wandb False")
    print("4. Evaluate the model after training:")
    print("   python evaluate.py --model_path ./model_output --visualize")
    print("5. Run inference on specific images:")
    print("   python demo.py --image_path path/to/image.png")


def main():
    """Main setup function"""
    print("\n=== Setting up OCR fine-tuning project for Google Colab ===\n")
    
    # Check GPU
    has_gpu = check_gpu()
    
    # Mount Google Drive
    drive_mounted = mount_drive()
    
    # Install dependencies
    deps_installed = install_dependencies()
    
    # Update config.py for Colab
    config_updated = update_config_for_colab()
    
    # Create directories in Google Drive if mounted
    if drive_mounted:
        dirs_created = create_drive_directories()
    else:
        dirs_created = False
        warnings.warn("Google Drive not mounted. Model checkpoints will not be saved persistently.")
    
    # Setup wandb
    wandb_setup = setup_wandb()
    
    # Display results
    print("\n=== Setup Results ===")
    print(f"GPU Available: {'✅' if has_gpu else '❌'}")
    print(f"Google Drive Mounted: {'✅' if drive_mounted else '❌'}")
    print(f"Dependencies Installed: {'✅' if deps_installed else '❌'}")
    print(f"Config Updated: {'✅' if config_updated else '❌'}")
    print(f"Drive Directories Created: {'✅' if dirs_created else '❌'}")
    print(f"Weights & Biases Setup: {'✅' if wandb_setup else '❌'}")
    
    # Display next steps
    display_next_steps()


if __name__ == "__main__":
    main() 