"""
Demo script for OCR model inference
"""

import os
import argparse
import torch
import matplotlib.pyplot as plt
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import time
import numpy as np
from pathlib import Path
import config


def parse_args():
    parser = argparse.ArgumentParser(description='Test OCR model on sample images')
    parser.add_argument('--model_path', type=str, default="./model_output",
                        help='Path to the fine-tuned model')
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to the image to recognize')
    parser.add_argument('--use_beam_search', action='store_true',
                        help='Use beam search for better results (slower)')
    parser.add_argument('--beam_size', type=int, default=5,
                        help='Beam size for beam search')
    return parser.parse_args()


def load_model(model_path):
    """Load the fine-tuned OCR model"""
    print(f"Loading model from {model_path}...")
    
    # Check if model path exists
    if not os.path.exists(model_path):
        print(f"Error: Model path {model_path} does not exist.")
        if os.path.exists("./model_output"):
            print("Using default model path ./model_output instead.")
            model_path = "./model_output"
        else:
            print(f"Error: Default model path does not exist either. Using pre-trained model {config.MODEL_NAME} instead.")
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


def preprocess_image(image_path):
    """Load and preprocess the image"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    try:
        # Open image
        image = Image.open(image_path).convert("RGB")
        return image
    except Exception as e:
        raise Exception(f"Error processing image: {e}")


def recognize_text(model, processor, image, use_beam_search=False, beam_size=5):
    """Recognize text in an image"""
    # Prepare device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    # Process image
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
    
    # Inference
    start_time = time.time()
    
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
    
    inference_time = time.time() - start_time
    
    # Decode prediction
    predicted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return predicted_text, inference_time


def display_result(image, predicted_text, inference_time):
    """Display the image and predicted text"""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(image)
    ax.axis('off')
    
    # Display prediction info
    title = f"Predicted: {predicted_text}\nInference time: {inference_time:.3f}s"
    plt.figtext(0.5, 0.01, title, wrap=True, horizontalalignment='center', fontsize=12)
    
    plt.tight_layout()
    plt.show()
    
    # Also print the result
    print(f"Predicted text: {predicted_text}")
    print(f"Inference time: {inference_time:.3f}s")


def main():
    # Parse arguments
    args = parse_args()
    
    # Load model and processor
    model, processor = load_model(args.model_path)
    
    # Load and preprocess image
    try:
        image = preprocess_image(args.image_path)
        
        # Recognize text
        predicted_text, inference_time = recognize_text(
            model, processor, image, 
            use_beam_search=args.use_beam_search,
            beam_size=args.beam_size
        )
        
        # Display result
        display_result(image, predicted_text, inference_time)
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main() 