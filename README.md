# Handwriting Recognition OCR Model Fine-Tuning

This project fine-tunes a transformer-based OCR model (TrOCR) for handwritten text recognition using the IAM and Imgur5K datasets.

## Project Structure

- `requirements.txt`: Required Python packages
- `config.py`: Configuration parameters for the project
- `data_utils.py`: Data loading and preprocessing utilities
- `model.py`: Model definition and fine-tuning utilities
- `train.py`: Training script with evaluation metrics
- `evaluate.py`: Evaluation script for the fine-tuned model
- `notebooks/`: Jupyter notebooks for exploration and demonstration

## Setup and Usage

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run training:
   ```
   python train.py
   ```

3. Evaluate model:
   ```
   python evaluate.py
   ```

## Model Details

The project uses Microsoft's TrOCR model (microsoft/trocr-large-handwritten) as the base model, which combines a Vision Transformer (ViT) encoder with a text Transformer decoder.

## Datasets

- IAM Handwriting Database: 13,353 handwritten English text lines from 657 writers
- Imgur5K: ~135K handwritten English words across 5K images 