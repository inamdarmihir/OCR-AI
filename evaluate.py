"""
Evaluation script for the OCR model.
"""

import os
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from model import calculate_cer, calculate_wer
import config
from data_utils import get_test_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate OCR model')
    parser.add_argument('--model_path', type=str, default="./model_output",
                        help='Path to the fine-tuned model')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for evaluation')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of samples to evaluate. If None, evaluate all samples.')
    parser.add_argument('--use_beam_search', action='store_true',
                        help='Use beam search for better results (slower)')
    parser.add_argument('--beam_size', type=int, default=5,
                        help='Beam size for beam search')
    parser.add_argument('--save_results', action='store_true',
                        help='Save evaluation results to a CSV file')
    parser.add_argument('--output_dir', type=str, default="./evaluation_results",
                        help='Directory to save evaluation results')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize sample predictions')
    parser.add_argument('--num_visualize', type=int, default=5,
                        help='Number of samples to visualize')
    return parser.parse_args()


def load_model(model_path):
    """Load the fine-tuned OCR model"""
    print(f"Loading model from {model_path}...")
    
    # Check if model path exists
    if not os.path.exists(model_path):
        print(f"Error: Model path {model_path} does not exist.")
        print(f"Falling back to pre-trained model {config.MODEL_NAME}")
        model_path = config.MODEL_NAME
    
    try:
        # Try to load the fine-tuned model
        model = VisionEncoderDecoderModel.from_pretrained(model_path)
        processor = TrOCRProcessor.from_pretrained(model_path)
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        print(f"Falling back to pre-trained model {config.MODEL_NAME}")
        model = VisionEncoderDecoderModel.from_pretrained(config.MODEL_NAME)
        processor = TrOCRProcessor.from_pretrained(config.MODEL_NAME)
    
    return model, processor


def evaluate_model(model, processor, dataloader, use_beam_search=False, beam_size=5, num_samples=None):
    """Evaluate the model on the test dataset"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_images = []
    
    # Check for mixed precision support
    use_amp = config.USE_MIXED_PRECISION and torch.cuda.is_available() and hasattr(torch.cuda, 'amp')
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    print(f"Evaluating model on {device} device...")
    print(f"Beam search: {'Enabled (beam size: ' + str(beam_size) + ')' if use_beam_search else 'Disabled'}")
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            if num_samples is not None and i * dataloader.batch_size >= num_samples:
                break
                
            images = batch["image"]
            texts = batch["text"]
            all_images.extend(images)
            all_targets.extend(texts)
            
            # Process images
            pixel_values = processor(images, return_tensors="pt").pixel_values.to(device)
            
            # Generate predictions
            if use_amp and torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    if use_beam_search:
                        generated_ids = model.generate(
                            pixel_values,
                            max_length=64,
                            num_beams=beam_size,
                            early_stopping=True
                        )
                    else:
                        generated_ids = model.generate(
                            pixel_values,
                            max_length=64
                        )
            else:
                if use_beam_search:
                    generated_ids = model.generate(
                        pixel_values,
                        max_length=64,
                        num_beams=beam_size,
                        early_stopping=True
                    )
                else:
                    generated_ids = model.generate(
                        pixel_values,
                        max_length=64
                    )
            
            # Decode predictions
            predicted_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
            all_predictions.extend(predicted_texts)
    
    # Calculate metrics
    cer = calculate_cer(all_predictions, all_targets)
    wer = calculate_wer(all_predictions, all_targets)
    
    results = {
        "cer": cer,
        "wer": wer,
        "predictions": all_predictions,
        "targets": all_targets,
        "images": all_images
    }
    
    return results


def visualize_results(images, predictions, targets, num_samples=5):
    """Visualize sample predictions"""
    num_samples = min(num_samples, len(images))
    indices = np.random.choice(len(images), num_samples, replace=False)
    
    fig, axs = plt.subplots(num_samples, 1, figsize=(12, 5 * num_samples))
    
    if num_samples == 1:
        axs = [axs]
    
    for i, idx in enumerate(indices):
        image = images[idx]
        prediction = predictions[idx]
        target = targets[idx]
        
        axs[i].imshow(image)
        axs[i].axis('off')
        axs[i].set_title(f"Predicted: {prediction}\nActual: {target}")
    
    plt.tight_layout()
    plt.show()


def save_results(results, output_dir):
    """Save evaluation results to a CSV file"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save metrics
    metrics = {
        "CER": [results["cer"]],
        "WER": [results["wer"]]
    }
    metrics_df = pd.DataFrame(metrics)
    metrics_file = os.path.join(output_dir, "metrics.csv")
    metrics_df.to_csv(metrics_file, index=False)
    print(f"Metrics saved to {metrics_file}")
    
    # Save predictions
    predictions_data = {
        "Target": results["targets"],
        "Prediction": results["predictions"]
    }
    predictions_df = pd.DataFrame(predictions_data)
    predictions_file = os.path.join(output_dir, "predictions.csv")
    predictions_df.to_csv(predictions_file, index=False)
    print(f"Predictions saved to {predictions_file}")


def main():
    # Parse arguments
    args = parse_args()
    
    # Load model and processor
    model, processor = load_model(args.model_path)
    
    # Create test dataloader
    test_dataloader = get_test_dataloader(processor, batch_size=args.batch_size)
    
    # Evaluate model
    results = evaluate_model(
        model, 
        processor, 
        test_dataloader, 
        use_beam_search=args.use_beam_search,
        beam_size=args.beam_size,
        num_samples=args.num_samples
    )
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Character Error Rate (CER): {results['cer']:.4f}")
    print(f"Word Error Rate (WER): {results['wer']:.4f}")
    
    # Visualize results
    if args.visualize:
        visualize_results(results["images"], results["predictions"], results["targets"], args.num_visualize)
    
    # Save results
    if args.save_results:
        save_results(results, args.output_dir)


if __name__ == "__main__":
    main() 