"""
Training script for fine-tuning the OCR model.
"""

import os
import time
import datetime
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from transformers import TrOCRProcessor, get_linear_schedule_with_warmup
from tqdm import tqdm
import wandb
import numpy as np
import random
import gc

import config
from data_utils import get_data_loaders, get_processor
from model import get_model, calculate_cer, calculate_wer, save_model, load_checkpoint


def set_seed(seed):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, dataloader, optimizer, scheduler, device, scaler=None):
    """
    Train for one epoch.
    
    Args:
        model: Model to train
        dataloader: DataLoader for training data
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on
        scaler: Gradient scaler for mixed precision training
        
    Returns:
        float: Average loss for the epoch
    """
    model.train()
    total_loss = 0
    
    for step, batch in enumerate(tqdm(dataloader, desc="Training")):
        # Move batch to device
        pixel_values = batch.pixel_values.to(device)
        labels = batch.labels.to(device)
        
        # Forward pass with mixed precision if enabled
        if config.USE_MIXED_PRECISION and scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss
        else:
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
        
        # Scale loss if using gradient accumulation
        if config.ACCUMULATION_STEPS > 1:
            loss = loss / config.ACCUMULATION_STEPS
            
        # Backward pass with mixed precision if enabled
        if config.USE_MIXED_PRECISION and scaler is not None:
            scaler.scale(loss).backward()
            
            # Step if accumulated enough
            if (step + 1) % config.ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
        else:
            loss.backward()
            
            # Step if accumulated enough
            if (step + 1) % config.ACCUMULATION_STEPS == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        
        # Track loss
        total_loss += loss.item() * config.ACCUMULATION_STEPS
        
        # Log every LOGGING_STEPS
        if (step + 1) % config.LOGGING_STEPS == 0:
            if args.use_wandb:  # Only log to wandb if enabled
                wandb.log({
                    "train_loss": loss.item() * config.ACCUMULATION_STEPS,
                    "learning_rate": scheduler.get_last_lr()[0],
                    "step": step
                })
            
            print(f"Step {step+1}/{len(dataloader)}: Loss = {loss.item() * config.ACCUMULATION_STEPS:.4f}")
            
            # For CPU training, clear memory more aggressively
            if device.type == 'cpu':
                gc.collect()
            
            # For T4 in Colab, monitor memory usage
            if device.type == 'cuda':
                gpu_memory = torch.cuda.memory_allocated(device) / 1024**3  # Convert to GB
                print(f"GPU Memory Usage: {gpu_memory:.2f} GB")
    
    return total_loss / len(dataloader)


def evaluate(model, processor, dataloader, device):
    """
    Evaluate the model.
    
    Args:
        model: Model to evaluate
        processor: Processor for decoding
        dataloader: DataLoader for evaluation data
        device: Device to evaluate on
        
    Returns:
        tuple: (loss, cer, wer)
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move batch to device
            pixel_values = batch.pixel_values.to(device)
            labels = batch.labels.to(device)
            
            # Forward pass
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            
            # Track loss
            total_loss += loss.item()
            
            # Generate predictions - with beam search optimized for device
            beam_size = 2 if device.type == 'cpu' else 4
            generated_ids = model.generate(
                pixel_values=pixel_values,
                max_length=config.MAX_LENGTH,
                num_beams=beam_size
            )
            
            # Decode predictions and labels
            pred_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
            target_texts = processor.batch_decode(labels, skip_special_tokens=True)
            
            all_preds.extend(pred_texts)
            all_targets.extend(target_texts)
            
            # For CPU training, clear memory more aggressively
            if device.type == 'cpu':
                gc.collect()
    
    # Calculate metrics
    cer = calculate_cer(all_preds, all_targets)
    wer = calculate_wer(all_preds, all_targets)
    
    return total_loss / len(dataloader), cer, wer


def train(args):
    """
    Train the model.
    
    Args:
        args: Command-line arguments
    """
    # Set up directories
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    
    # For Google Colab, set up Drive save directory if enabled
    if hasattr(config, 'SAVE_TO_DRIVE') and config.SAVE_TO_DRIVE:
        if os.path.exists(config.DRIVE_MOUNT_PATH):
            os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
            print(f"Will save checkpoints to Google Drive at: {config.CHECKPOINT_DIR}")
        else:
            print("Google Drive not mounted, SAVE_TO_DRIVE will be ignored")
            config.SAVE_TO_DRIVE = False
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Initialize wandb if enabled
    if args.use_wandb:
        try:
            wandb.init(
                project="ocr-handwriting-recognition",
                name=f"trocr-finetuning-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
                config={
                    "model": config.MODEL_NAME,
                    "batch_size": config.BATCH_SIZE,
                    "learning_rate": config.LEARNING_RATE,
                    "epochs": config.NUM_EPOCHS,
                    "mixed_precision": config.USE_MIXED_PRECISION,
                    "augmentation": config.USE_AUGMENTATION,
                    "device": "cpu" if not torch.cuda.is_available() else "cuda",
                    "gradient_accumulation_steps": config.ACCUMULATION_STEPS
                }
            )
        except Exception as e:
            print(f"Failed to initialize wandb: {e}")
            print("Training will continue without wandb logging")
            args.use_wandb = False
    
    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Print GPU information if available
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Optimal settings for different GPUs
        if "T4" in torch.cuda.get_device_name(0):
            print("Detected T4 GPU - Using optimal settings for T4")
            if config.BATCH_SIZE > 4 and config.USE_MIXED_PRECISION:
                print("WARNING: Batch size > 4 may cause OOM on T4. Consider reducing batch size if you encounter memory issues.")
        
        if "K80" in torch.cuda.get_device_name(0):
            print("Detected K80 GPU - This GPU has less memory, adjusting settings")
            if config.BATCH_SIZE > 2:
                print("WARNING: Batch size > 2 may cause OOM on K80. Consider reducing batch size.")
            if not args.use_smaller_model:
                print("WARNING: Consider using a smaller model with the --use_smaller_model flag for K80 GPU.")
    else:
        # For CPU training, print warning
        print("WARNING: Training on CPU will be very slow. Consider using a GPU for faster training.")
        print(f"CPU Training optimizations: Batch size={config.BATCH_SIZE}, Gradient accumulation={config.ACCUMULATION_STEPS}")
        if hasattr(config, 'MAX_TRAIN_SAMPLES') and config.MAX_TRAIN_SAMPLES:
            print(f"Using limited dataset: {config.MAX_TRAIN_SAMPLES} training samples")
    
    # For Colab or sessions with time limits, estimate training time
    if args.estimate_time:
        if torch.cuda.is_available():
            if "T4" in torch.cuda.get_device_name(0):
                samples_per_second = 5  # Approximate for T4
            elif "K80" in torch.cuda.get_device_name(0): 
                samples_per_second = 2  # Approximate for K80
            else:
                samples_per_second = 4  # Default estimate
        else:
            samples_per_second = 0.2  # Very slow on CPU
        
        # Calculate approximate training time
        total_samples = config.MAX_TRAIN_SAMPLES if hasattr(config, 'MAX_TRAIN_SAMPLES') and config.MAX_TRAIN_SAMPLES else 13353  # IAM dataset size
        total_seconds = (total_samples / samples_per_second) * config.NUM_EPOCHS
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        
        print(f"Estimated training time: {hours:.0f} hours and {minutes:.0f} minutes")
        print("Note: Colab sessions disconnect after 12 hours of runtime")
        
        if hours > 11:
            print("WARNING: Estimated training time exceeds Colab's session limit!")
            print("Consider reducing MAX_TRAIN_SAMPLES or NUM_EPOCHS")
    
    # Use smaller model if specified (especially for Colab K80)
    if args.use_smaller_model:
        print("Using smaller model: microsoft/trocr-base-handwritten")
        config.MODEL_NAME = "microsoft/trocr-base-handwritten"
    
    # Get model and processor
    print("Loading model and processor...")
    model = get_model()
    processor = get_processor()
    print("Model and processor loaded.")
    
    # Move model to device
    model.to(device)
    
    # Enable mixed precision if specified and on GPU
    scaler = torch.cuda.amp.GradScaler() if config.USE_MIXED_PRECISION and torch.cuda.is_available() else None
    
    # Get data loaders
    print("Preparing datasets...")
    train_loader, val_loader, test_loader = get_data_loaders()
    print("Datasets prepared.")
    
    # Set up optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Set up scheduler
    total_steps = len(train_loader) * config.NUM_EPOCHS // config.ACCUMULATION_STEPS
    
    if args.scheduler == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.WARMUP_STEPS,
            num_training_steps=total_steps
        )
    else:  # one_cycle
        scheduler = OneCycleLR(
            optimizer,
            max_lr=config.LEARNING_RATE,
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=10000.0
        )
    
    # Load checkpoint if specified
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint.startswith(config.DRIVE_MOUNT_PATH):
            print(f"Loading checkpoint from Google Drive: {args.resume_from_checkpoint}")
        model = load_checkpoint(model, args.resume_from_checkpoint)
    
    # Multi-GPU training if available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    # Train
    best_cer = float('inf')
    
    # Save frequency - more frequent for CPU to handle interruptions
    save_every_epoch = device.type == 'cpu'
    
    # For Colab, save more frequently and to Drive if enabled
    if torch.cuda.is_available() and "T4" in torch.cuda.get_device_name(0):
        save_every_epoch = True
        print("Saving checkpoints every epoch to handle potential Colab disconnects")
    
    print("\n" + "="*50)
    print(f"Starting training with {config.NUM_EPOCHS} epochs")
    print(f"Batch size: {config.BATCH_SIZE}, Gradient accumulation steps: {config.ACCUMULATION_STEPS}")
    if hasattr(config, 'MAX_TRAIN_SAMPLES') and config.MAX_TRAIN_SAMPLES:
        print(f"Training on {config.MAX_TRAIN_SAMPLES} samples")
    print("="*50 + "\n")
    
    for epoch in range(config.NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
        
        # Train for one epoch
        start_time = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, scaler)
        
        # Evaluate
        val_loss, val_cer, val_wer = evaluate(
            model.module if hasattr(model, "module") else model,
            processor,
            val_loader,
            device
        )
        
        # Log metrics
        elapsed_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS} - "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val CER: {val_cer:.4f}, "
              f"Val WER: {val_wer:.4f}, "
              f"Time: {elapsed_time:.2f}s")
        
        if args.use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_cer": val_cer,
                "val_wer": val_wer,
                "epoch_time": elapsed_time
            })
        
        # Save best model
        if val_cer < best_cer:
            best_cer = val_cer
            model_to_save = model.module if hasattr(model, "module") else model
            
            # Save to local directory
            save_model(
                model_to_save,
                processor,
                os.path.join(config.MODEL_SAVE_DIR, f"best_model")
            )
            print(f"Saved best model with CER: {best_cer:.4f}")
            
            # Also save to Drive if enabled (for Colab)
            if hasattr(config, 'SAVE_TO_DRIVE') and config.SAVE_TO_DRIVE:
                try:
                    save_model(
                        model_to_save,
                        processor,
                        os.path.join(config.CHECKPOINT_DIR, f"best_model")
                    )
                    print(f"Saved best model to Drive")
                except Exception as e:
                    print(f"Failed to save to Drive: {e}")
        
        # Save checkpoint every epoch for CPU training or Colab to handle interruptions
        if save_every_epoch or (epoch + 1) % args.save_every == 0:
            model_to_save = model.module if hasattr(model, "module") else model
            
            # Save to local directory
            save_model(
                model_to_save,
                processor,
                os.path.join(config.MODEL_SAVE_DIR, f"checkpoint-epoch-{epoch+1}")
            )
            print(f"Saved checkpoint for epoch {epoch+1}")
            
            # Also save to Drive if enabled (for Colab)
            if hasattr(config, 'SAVE_TO_DRIVE') and config.SAVE_TO_DRIVE:
                try:
                    save_model(
                        model_to_save,
                        processor,
                        os.path.join(config.CHECKPOINT_DIR, f"checkpoint-epoch-{epoch+1}")
                    )
                    print(f"Saved checkpoint to Drive")
                except Exception as e:
                    print(f"Failed to save to Drive: {e}")
        
        # Force memory cleanup after each epoch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Print GPU memory usage
            memory_allocated = torch.cuda.memory_allocated(device) / 1e9  # in GB
            memory_reserved = torch.cuda.memory_reserved(device) / 1e9  # in GB
            print(f"GPU Memory: {memory_allocated:.2f} GB allocated, {memory_reserved:.2f} GB reserved")
    
    # Final evaluation on test set
    print("Evaluating on test set...")
    model_to_eval = model.module if hasattr(model, "module") else model
    test_loss, test_cer, test_wer = evaluate(model_to_eval, processor, test_loader, device)
    print(f"Test Loss: {test_loss:.4f}, Test CER: {test_cer:.4f}, Test WER: {test_wer:.4f}")
    
    if args.use_wandb:
        wandb.log({
            "test_loss": test_loss,
            "test_cer": test_cer,
            "test_wer": test_wer
        })
        wandb.finish()
    
    # Save final model
    model_to_save = model.module if hasattr(model, "module") else model
    
    # Save to local directory
    save_model(
        model_to_save,
        processor,
        os.path.join(config.MODEL_SAVE_DIR, "final_model")
    )
    print("Training complete!")
    
    # Also save to Drive if enabled (for Colab)
    if hasattr(config, 'SAVE_TO_DRIVE') and config.SAVE_TO_DRIVE:
        try:
            save_model(
                model_to_save,
                processor,
                os.path.join(config.CHECKPOINT_DIR, "final_model")
            )
            print(f"Saved final model to Drive")
        except Exception as e:
            print(f"Failed to save to Drive: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train OCR model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases for logging")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--save_every", type=int, default=1, help="Save checkpoint every N epochs")
    parser.add_argument("--scheduler", type=str, default="linear", choices=["linear", "one_cycle"], help="Scheduler type")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with minimal data")
    parser.add_argument("--estimate_time", action="store_true", help="Estimate training time before starting")
    parser.add_argument("--use_smaller_model", action="store_true", help="Use smaller trocr-base model instead of trocr-large")
    
    args = parser.parse_args()
    
    # If debug mode is enabled, override some config settings
    if args.debug:
        config.MAX_TRAIN_SAMPLES = 10
        config.MAX_VAL_SAMPLES = 5
        config.MAX_TEST_SAMPLES = 5
        config.NUM_EPOCHS = 2
    
    train(args) 