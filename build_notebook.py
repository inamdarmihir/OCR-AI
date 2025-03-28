import json

# Define the notebook structure
notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {"id": "intro"},
            "source": [
                "# OCR Model Fine-Tuning on Google Colab\n",
                "\n",
                "This notebook walks through the process of fine-tuning a TrOCR model on the IAM Handwriting dataset using Google Colab's T4 GPU.\n",
                "\n",
                "**Project Overview:** We'll fine-tune a vision-encoder-decoder model for optical character recognition (OCR) on handwritten text. The model combines a vision transformer (ViT) as encoder and a language model as decoder.\n",
                "\n",
                "**Make sure you have GPU acceleration enabled!**\n",
                "To check: Runtime > Change runtime type > Hardware accelerator > GPU"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"id": "check_gpu"},
            "source": [
                "# Check for GPU availability\n",
                "!nvidia-smi\n",
                "\n",
                "import torch\n",
                "print(f\"PyTorch version: {torch.__version__}\")\n",
                "print(f\"CUDA available: {torch.cuda.is_available()}\")\n",
                "if torch.cuda.is_available():\n",
                "    device_name = torch.cuda.get_device_name(0)\n",
                "    print(f\"GPU: {device_name}\")\n",
                "    \n",
                "    # Print total GPU memory\n",
                "    total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Convert to GB\n",
                "    print(f\"Total GPU memory: {total_memory:.2f} GB\")\n",
                "else:\n",
                "    print(\"No GPU available. Please enable GPU acceleration in Runtime > Change runtime type.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {"id": "mount_drive_header"},
            "source": [
                "## Step 1: Mount Google Drive\n",
                "\n",
                "Mount your Google Drive to save model checkpoints and results. This ensures your trained model persists after the Colab session ends."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"id": "mount_drive"},
            "source": [
                "from google.colab import drive\n",
                "drive.mount('/content/drive')\n",
                "\n",
                "# Create directories for saving model outputs\n",
                "!mkdir -p /content/drive/MyDrive/ocr_output\n",
                "!mkdir -p /content/drive/MyDrive/ocr_output/checkpoints\n",
                "!mkdir -p /content/drive/MyDrive/ocr_output/final_model\n",
                "!mkdir -p /content/drive/MyDrive/ocr_output/evaluation\n",
                "\n",
                "print(\"Google Drive mounted. Model checkpoints and results will be saved to /content/drive/MyDrive/ocr_output\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {"id": "setup_code_header"},
            "source": [
                "## Step 2: Get the Code\n",
                "\n",
                "There are two options to get the code:\n",
                "\n",
                "1. Clone from GitHub repository (preferred)\n",
                "2. Upload required files directly to Colab"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"id": "clone_repo"},
            "source": [
                "# Option 1: Clone from GitHub\n",
                "# Replace with your actual repository URL\n",
                "!git clone https://github.com/yourusername/ocr-finetuning.git\n",
                "# Change to project directory\n",
                "%cd ocr-finetuning"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {"id": "upload_files_header"},
            "source": [
                "**Option 2: Upload files directly**\n",
                "\n",
                "If you don't have a GitHub repository, you can upload the files directly to Colab. You'll need to upload the following files:\n",
                "- `config.py`: Configuration settings\n",
                "- `data_utils.py`: Data handling utilities\n",
                "- `model.py`: Model definition\n",
                "- `train.py`: Training script\n",
                "- `evaluate.py`: Evaluation script\n",
                "- `demo.py`: Inference script\n",
                "- `setup_colab.py`: Colab setup script\n",
                "- `verify_setup.py`: Verification script\n",
                "- `run_test.py`: Test script\n",
                "\n",
                "**IMPORTANT**: Only run the code cell below if you're uploading files manually. If you cloned from GitHub, skip this cell."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"id": "upload_files"},
            "source": [
                "# Option 2: Upload files directly (only run if you didn't clone from GitHub)\n",
                "# This will allow you to upload files\n",
                "from google.colab import files\n",
                "\n",
                "print(\"Please upload the project files. You can select multiple files at once.\")\n",
                "uploaded = files.upload()\n",
                "\n",
                "print(\"Files uploaded:\")\n",
                "for filename in uploaded.keys():\n",
                "    print(f\"- {filename}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {"id": "install_deps_header"},
            "source": [
                "## Step 3: Install Dependencies\n",
                "\n",
                "Install the required packages for the project. We'll use the setup_colab.py script to handle this."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"id": "install_dependencies"},
            "source": [
                "!python setup_colab.py"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {"id": "verify_setup_header"},
            "source": [
                "## Step 4: Verify Setup\n",
                "\n",
                "Make sure everything is properly set up before proceeding with training."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"id": "verify_setup"},
            "source": [
                "!python verify_setup.py"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {"id": "quick_test_header"},
            "source": [
                "## Step 5: Run a Quick Test\n",
                "\n",
                "Before starting the full training, run a quick test with a small subset of data to make sure everything works."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"id": "run_quick_test"},
            "source": [
                "!python run_test.py --num_samples 5 --batch_size 2 --num_epochs 1"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {"id": "training_header"},
            "source": [
                "## Step 6: Start Training\n",
                "\n",
                "Now that we've verified everything is working, we can start the full training process.\n",
                "\n",
                "**Note**: You may want to edit `config.py` to adjust training parameters for your needs before starting full training."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"id": "edit_config"},
            "source": [
                "# View current config\n",
                "!cat config.py"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"id": "run_training"},
            "source": [
                "# Start training\n",
                "# Set use_wandb to True if you want to track the training with Weights & Biases\n",
                "!python train.py --use_wandb False"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {"id": "evaluation_header"},
            "source": [
                "## Step 7: Evaluate the Model\n",
                "\n",
                "After training, evaluate the model to assess its performance."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"id": "evaluate_model"},
            "source": [
                "!python evaluate.py --model_path ./model_output --visualize --save_results"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {"id": "demo_header"},
            "source": [
                "## Step 8: Run the Demo\n",
                "\n",
                "Test the model on some sample images to see how it performs on real-world tasks."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"id": "download_sample_image"},
            "source": [
                "# Download a sample image from IAM dataset\n",
                "!mkdir -p sample_images\n",
                "!wget https://fki.tic.heia-fr.ch/static/img/a01-122-02.jpg -O sample_images/sample1.jpg\n",
                "\n",
                "# Display the image\n",
                "from PIL import Image\n",
                "import matplotlib.pyplot as plt\n",
                "\n",
                "img = Image.open('sample_images/sample1.jpg')\n",
                "plt.figure(figsize=(10, 4))\n",
                "plt.imshow(img)\n",
                "plt.axis('off')\n",
                "plt.title('Sample Image')\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"id": "run_demo"},
            "source": [
                "# Run demo on the sample image\n",
                "!python demo.py --image_path sample_images/sample1.jpg --use_beam_search"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {"id": "colab_tips_header"},
            "source": [
                "## Colab Tips\n",
                "\n",
                "### Preventing Disconnection\n",
                "Google Colab may disconnect after a period of inactivity. To prevent this, you can use the code below which keeps the session active.\n",
                "\n",
                "**Note**: Only use this when running long training sessions and when you're actively monitoring the notebook, as it consumes resources.\n",
                "\n",
                "### Managing RAM\n",
                "If you're experiencing out-of-memory errors, you can clear the runtime memory with the following code.\n",
                "\n",
                "### Session Duration\n",
                "Remember that Colab sessions have a limited duration (usually 12 hours). Save your model checkpoints to Google Drive regularly."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"id": "prevent_disconnect"},
            "source": [
                "# Prevent Colab from disconnecting due to inactivity\n",
                "# Only run this cell when necessary (during long training runs)\n",
                "\n",
                "from IPython.display import display, Javascript\n",
                "import time\n",
                "\n",
                "def keep_alive():\n",
                "    display(Javascript('''\n",
                "    function click() {\n",
                "        document.querySelector(\"colab-toolbar-button#connect\").click()\n",
                "    }\n",
                "    setInterval(click, 60000)\n",
                "    '''))\n",
                "\n",
                "keep_alive()  # This will automatically click the \"Connect\" button every 60 seconds"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"id": "clear_memory"},
            "source": [
                "# Clear memory if needed\n",
                "import gc\n",
                "import torch\n",
                "\n",
                "gc.collect()\n",
                "torch.cuda.empty_cache()\n",
                "print(\"Memory cleared!\")"
            ]
        }
    ],
    "metadata": {
        "accelerator": "GPU",
        "colab": {
            "name": "OCR_Training_Colab.ipynb",
            "provenance": [],
            "gpuType": "T4"
        },
        "kernelspec": {
            "display_name": "Python 3",
            "name": "python3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 0
}

# Write the notebook to a file
with open('OCR_Training_Colab.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)

print("Notebook successfully created: OCR_Training_Colab.ipynb") 