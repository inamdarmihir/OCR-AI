import json

# Define the notebook structure
notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "# OCR Model Fine-Tuning on Google Colab\n\nThis notebook walks through the process of fine-tuning a TrOCR model on the IAM Handwriting dataset using Google Colab's T4 GPU.\n\n**Project Overview:** We'll fine-tune a vision-encoder-decoder model for optical character recognition (OCR) on handwritten text. The model combines a vision transformer (ViT) as encoder and a language model as decoder.\n\n**Make sure you have GPU acceleration enabled!**\nTo check: Runtime > Change runtime type > Hardware accelerator > GPU"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": "# Check for GPU availability\n!nvidia-smi\n\nimport torch\nprint(f\"PyTorch version: {torch.__version__}\")\nprint(f\"CUDA available: {torch.cuda.is_available()}\")\nif torch.cuda.is_available():\n    device_name = torch.cuda.get_device_name(0)\n    print(f\"GPU: {device_name}\")\n    \n    # Print total GPU memory\n    total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Convert to GB\n    print(f\"Total GPU memory: {total_memory:.2f} GB\")\nelse:\n    print(\"No GPU available. Please enable GPU acceleration in Runtime > Change runtime type.\")"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "## Step 1: Mount Google Drive\n\nMount your Google Drive to save model checkpoints and results. This ensures your trained model persists after the Colab session ends."
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": "from google.colab import drive\ndrive.mount('/content/drive')\n\n# Create directories for saving model outputs\n!mkdir -p /content/drive/MyDrive/ocr_output\n!mkdir -p /content/drive/MyDrive/ocr_output/checkpoints\n!mkdir -p /content/drive/MyDrive/ocr_output/final_model\n!mkdir -p /content/drive/MyDrive/ocr_output/evaluation\n\nprint(\"Google Drive mounted. Model checkpoints and results will be saved to /content/drive/MyDrive/ocr_output\")"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "## Step 2: Get the Code\n\nThere are two options to get the code:\n\n1. Clone from GitHub repository (preferred)\n2. Upload required files directly to Colab"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": "# Option 1: Clone from GitHub\n# Replace with your actual repository URL\n!git clone https://github.com/yourusername/ocr-finetuning.git\n# Change to project directory\n%cd ocr-finetuning"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "**Option 2: Upload files directly**\n\nIf you don't have a GitHub repository, you can upload the files directly to Colab. You'll need to upload the following files:\n- `config.py`: Configuration settings\n- `data_utils.py`: Data handling utilities\n- `model.py`: Model definition\n- `train.py`: Training script\n- `evaluate.py`: Evaluation script\n- `demo.py`: Inference script\n- `setup_colab.py`: Colab setup script\n- `verify_setup.py`: Verification script\n- `run_test.py`: Test script\n\n**IMPORTANT**: Only run the code cell below if you're uploading files manually. If you cloned from GitHub, skip this cell."
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": "# Option 2: Upload files directly (only run if you didn't clone from GitHub)\n# This will allow you to upload files\nfrom google.colab import files\n\nprint(\"Please upload the project files. You can select multiple files at once.\")\nuploaded = files.upload()\n\nprint(\"Files uploaded:\")\nfor filename in uploaded.keys():\n    print(f\"- {filename}\")"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "## Step 3: Install Dependencies\n\nInstall the required packages for the project. We'll use the setup_colab.py script to handle this."
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": "!python setup_colab.py"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "## Step 4: Verify Setup\n\nMake sure everything is properly set up before proceeding with training."
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": "!python verify_setup.py"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "## Step 5: Run a Quick Test\n\nBefore starting the full training, run a quick test with a small subset of data to make sure everything works."
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": "!python run_test.py --num_samples 5 --batch_size 2 --num_epochs 1"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "## Step 6: Start Training\n\nNow that we've verified everything is working, we can start the full training process.\n\n**Note**: You may want to edit `config.py` to adjust training parameters for your needs before starting full training."
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": "# View current config\n!cat config.py"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": "# Start training\n# Set use_wandb to True if you want to track the training with Weights & Biases\n!python train.py --use_wandb False"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "## Step 7: Evaluate the Model\n\nAfter training, evaluate the model to assess its performance."
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": "!python evaluate.py --model_path ./model_output --visualize --save_results"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "## Step 8: Run the Demo\n\nTest the model on some sample images to see how it performs on real-world tasks."
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": "# Download a sample image from IAM dataset\n!mkdir -p sample_images\n!wget https://fki.tic.heia-fr.ch/static/img/a01-122-02.jpg -O sample_images/sample1.jpg\n\n# Display the image\nfrom PIL import Image\nimport matplotlib.pyplot as plt\n\nimg = Image.open('sample_images/sample1.jpg')\nplt.figure(figsize=(10, 4))\nplt.imshow(img)\nplt.axis('off')\nplt.title('Sample Image')\nplt.show()"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": "# Run demo on the sample image\n!python demo.py --image_path sample_images/sample1.jpg --use_beam_search"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "## Colab Tips\n\n### Preventing Disconnection\nGoogle Colab may disconnect after a period of inactivity. To prevent this, you can use the code below which keeps the session active.\n\n**Note**: Only use this when running long training sessions and when you're actively monitoring the notebook, as it consumes resources.\n\n### Managing RAM\nIf you're experiencing out-of-memory errors, you can clear the runtime memory with the following code.\n\n### Session Duration\nRemember that Colab sessions have a limited duration (usually 12 hours). Save your model checkpoints to Google Drive regularly."
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": "# Prevent Colab from disconnecting due to inactivity\n# Only run this cell when necessary (during long training runs)\n\nfrom IPython.display import display, Javascript\nimport time\n\ndef keep_alive():\n    display(Javascript('''\n    function click() {\n        document.querySelector(\"colab-toolbar-button#connect\").click()\n    }\n    setInterval(click, 60000)\n    '''))\n\nkeep_alive()  # This will automatically click the \"Connect\" button every 60 seconds"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": "# Clear memory if needed\nimport gc\nimport torch\n\ngc.collect()\ntorch.cuda.empty_cache()\nprint(\"Memory cleared!\")"
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

# Using a simpler approach with string instead of array for source
for cell in notebook["cells"]:
    if isinstance(cell["source"], str):
        # Already a string, no need to modify
        pass
    elif isinstance(cell["source"], list):
        # Join the list into a string
        cell["source"] = "".join(cell["source"])

# Write the notebook to a file
with open('OCR_Training_Colab.ipynb', 'w') as f:
    json.dump(notebook, f)

print("Notebook successfully created: OCR_Training_Colab.ipynb") 