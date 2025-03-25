"""
Quick test script to validate the OCR model training process.
Runs a minimal training process on a small subset of data.
"""

import os
import argparse
import torch
import time
from pathlib import Path
import config
import numpy as np
from transformers import TrOCRProcessor
from model import get_model
from data_utils import get_data_loaders
from train import train_model


def parse_args():
    parser = argparse.ArgumentParser(description="Run a quick test of OCR model training")
    parser.add_argument("--num_samples", type=int, default=10, 
                        help="Number of samples to use for training")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=1,
                        help="Number of epochs for training")
    parser.add_argument("--save_model", action="store_true",
                        help="Whether to save the model after training")
    parser.add_argument("--output_dir", type=str, default="./test_output",
                        help="Directory to save the model")
    return parser.parse_args()


def setup_test_config(args):
    """Temporarily modify config settings for quick test"""
    # Store original values to restore later
    original_values = {
        "NUM_TRAIN_SAMPLES": getattr(config, "NUM_TRAIN_SAMPLES", None),
        "NUM_VAL_SAMPLES": getattr(config, "NUM_VAL_SAMPLES", None),
        "NUM_TEST_SAMPLES": getattr(config, "NUM_TEST_SAMPLES", None),
        "BATCH_SIZE": config.BATCH_SIZE,
        "NUM_EPOCHS": config.NUM_EPOCHS,
        "SAVE_CHECKPOINT_STEPS": config.SAVE_CHECKPOINT_STEPS,
        "LOG_STEPS": config.LOG_STEPS
    }
    
    # Set test values
    config.NUM_TRAIN_SAMPLES = args.num_samples
    config.NUM_VAL_SAMPLES = args.num_samples // 2
    config.NUM_TEST_SAMPLES = args.num_samples // 2
    config.BATCH_SIZE = args.batch_size
    config.NUM_EPOCHS = args.num_epochs
    config.SAVE_CHECKPOINT_STEPS = 5
    config.LOG_STEPS = 2
    
    return original_values


def restore_config(original_values):
    """Restore original config values"""
    for key, value in original_values.items():
        if value is not None:
            setattr(config, key, value)


def run_test(args):
    """Run a quick test of the training process"""
    start_time = time.time()
    
    print("\n=== Running Quick Test of OCR Training Process ===\n")
    
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Temporarily modify config for quick test
    original_values = setup_test_config(args)
    
    # Print test configuration
    print("\nTest Configuration:")
    print(f"- Number of training samples: {config.NUM_TRAIN_SAMPLES}")
    print(f"- Number of validation samples: {config.NUM_VAL_SAMPLES}")
    print(f"- Batch size: {config.BATCH_SIZE}")
    print(f"- Number of epochs: {config.NUM_EPOCHS}")
    
    try:
        # Get processor and model
        processor = TrOCRProcessor.from_pretrained(config.MODEL_NAME)
        model = get_model()
        
        # Get data loaders
        train_loader, val_loader, test_loader = get_data_loaders(processor)
        
        # Check if data loaders are properly set up
        print("\nDataset sizes:")
        print(f"- Training samples: {len(train_loader.dataset)}")
        print(f"- Validation samples: {len(val_loader.dataset)}")
        
        # Run a quick training process
        print("\nStarting test training process...")
        train_results = train_model(model, processor, train_loader, val_loader, use_wandb=False)
        
        # Create output directory if saving model
        if args.save_model:
            os.makedirs(args.output_dir, exist_ok=True)
            model.save_pretrained(args.output_dir)
            processor.save_pretrained(args.output_dir)
            print(f"\nModel saved to {args.output_dir}")
        
        print("\n✅ Test completed successfully!")
        print(f"Final training loss: {train_results['train_loss']:.4f}")
        print(f"Final validation loss: {train_results['val_loss']:.4f}")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Restore original config values
        restore_config(original_values)
    
    elapsed_time = time.time() - start_time
    print(f"\nTest completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")


def main():
    args = parse_args()
    run_test(args)


if __name__ == "__main__":
    main() 