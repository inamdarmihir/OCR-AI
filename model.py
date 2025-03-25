"""
Model definition and utilities for OCR fine-tuning.
"""

import os
import torch
import torch.nn as nn
from transformers import TrOCRForCausalLM, VisionEncoderDecoderModel
import config


def get_model():
    """
    Get the TrOCR model for fine-tuning.
    
    Returns:
        TrOCRForCausalLM: The TrOCR model
    """
    print(f"Loading model {config.MODEL_NAME}...")
    model = VisionEncoderDecoderModel.from_pretrained(config.MODEL_NAME)
    
    # Configure training parameters
    model.config.decoder_start_token_id = 0
    model.config.pad_token_id = 0
    
    # Enable gradient checkpointing for memory efficiency
    model.encoder.config.use_cache = False
    model.decoder.config.use_cache = False
    
    # Device-specific optimizations
    if torch.cuda.is_available():
        # GPU optimizations
        device_name = torch.cuda.get_device_name(0)
        print(f"Optimizing model for {device_name}")
        
        # Enable gradient checkpointing for all GPUs to save memory
        if hasattr(model.encoder, "gradient_checkpointing_enable"):
            model.encoder.gradient_checkpointing_enable()
            print("Enabled gradient checkpointing for encoder")
        if hasattr(model.decoder, "gradient_checkpointing_enable"):
            model.decoder.gradient_checkpointing_enable()
            print("Enabled gradient checkpointing for decoder")
            
        # Special optimization for T4 GPU (most common in Colab)
        if "T4" in device_name:
            # T4 has 16GB VRAM - Use efficient attention if available
            if hasattr(model.encoder, "config"):
                # Some encoder models support memory-efficient attention mechanisms
                # that make better use of T4 memory
                if hasattr(model.encoder.config, "attention_mode"):
                    model.encoder.config.attention_mode = "xformers"
                    print("Enabled xformers efficient attention for encoder")
    elif hasattr(config, 'CPU_TRAINING') and config.CPU_TRAINING:
        # CPU optimizations
        print("Optimizing model for CPU training")
        
        # Enable gradient checkpointing for memory efficiency
        if hasattr(model.encoder, "gradient_checkpointing_enable"):
            model.encoder.gradient_checkpointing_enable()
        if hasattr(model.decoder, "gradient_checkpointing_enable"):
            model.decoder.gradient_checkpointing_enable()
            
        print("Enabled CPU-specific optimizations for model")
    
    return model


def calculate_cer(pred_texts, target_texts):
    """
    Calculate Character Error Rate (CER).
    
    Args:
        pred_texts (list): List of predicted texts
        target_texts (list): List of target texts
        
    Returns:
        float: CER score
    """
    total_cer = 0
    total_chars = 0
    
    for pred, target in zip(pred_texts, target_texts):
        # Calculate edit distance
        distance = levenshtein_distance(pred, target)
        total_cer += distance
        total_chars += len(target)
    
    if total_chars == 0:
        return 0.0
        
    return total_cer / total_chars


def calculate_wer(pred_texts, target_texts):
    """
    Calculate Word Error Rate (WER).
    
    Args:
        pred_texts (list): List of predicted texts
        target_texts (list): List of target texts
        
    Returns:
        float: WER score
    """
    total_wer = 0
    total_words = 0
    
    for pred, target in zip(pred_texts, target_texts):
        # Split into words
        pred_words = pred.split()
        target_words = target.split()
        
        # Calculate edit distance
        distance = levenshtein_distance(pred_words, target_words)
        total_wer += distance
        total_words += len(target_words)
    
    if total_words == 0:
        return 0.0
        
    return total_wer / total_words


def levenshtein_distance(s1, s2):
    """
    Calculate Levenshtein distance between two sequences.
    
    Args:
        s1: First sequence
        s2: Second sequence
        
    Returns:
        int: Levenshtein distance
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Calculate insertions, deletions, and substitutions
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            
            # Get minimum
            current_row.append(min(insertions, deletions, substitutions))
        
        # Update previous row
        previous_row = current_row
    
    return previous_row[-1]


def save_model(model, tokenizer, output_dir):
    """
    Save model and tokenizer.
    
    Args:
        model: Model to save
        tokenizer: Tokenizer to save
        output_dir: Directory to save to
    """
    # Create directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Saving model to {output_dir}...")
    
    try:
        # Save model
        model.save_pretrained(output_dir)
        
        # Save tokenizer
        tokenizer.save_pretrained(output_dir)
        
        print(f"Model and tokenizer saved to {output_dir}")
        return True
    except Exception as e:
        print(f"Error saving model to {output_dir}: {e}")
        return False


def load_checkpoint(model, checkpoint_path):
    """
    Load model checkpoint.
    
    Args:
        model: Model to load checkpoint into
        checkpoint_path: Path to checkpoint
        
    Returns:
        model: Model with loaded checkpoint
    """
    try:
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}...")
            
            # Handle Google Drive paths on Colab that might have spaces
            if ' ' in checkpoint_path:
                print("Warning: Checkpoint path contains spaces, which might cause issues")
            
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(state_dict)
            print(f"Loaded checkpoint from {checkpoint_path}")
        else:
            print(f"No checkpoint found at {checkpoint_path}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Continuing with initial model weights")
    
    return model 