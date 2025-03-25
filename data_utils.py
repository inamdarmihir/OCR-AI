"""
Data loading and preprocessing utilities for OCR fine-tuning.
"""

import os
import cv2
import numpy as np
import random
from PIL import Image, ImageOps, ImageFilter
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import TrOCRProcessor
import config


def get_processor():
    """
    Get the TrOCR processor for image and text tokenization.
    """
    return TrOCRProcessor.from_pretrained(config.MODEL_NAME)


class ImageAugmentation:
    """
    Image augmentation for OCR data.
    """
    def __init__(self, prob=config.AUGMENTATION_PROBABILITY):
        self.prob = prob
        
    def apply(self, image):
        """
        Apply random augmentations to the image.
        
        Args:
            image (PIL.Image): Input image
            
        Returns:
            PIL.Image: Augmented image
        """
        if not config.USE_AUGMENTATION:
            return image
            
        # Convert to PIL Image if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.array(image))
            
        # Random rotation (slight)
        if random.random() < self.prob:
            angle = random.uniform(-2, 2)
            image = image.rotate(angle, expand=False, fillcolor=(255, 255, 255))
            
        # Random brightness/contrast adjustment
        if random.random() < self.prob:
            enhancer = ImageOps.autocontrast
            image = enhancer(image)
            
        # Add slight noise
        if random.random() < self.prob:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 0.5)))
            
        # Random scaling
        if random.random() < self.prob:
            scale = random.uniform(0.95, 1.05)
            width, height = image.size
            new_width, new_height = int(width * scale), int(height * scale)
            image = image.resize((new_width, new_height), Image.LANCZOS)
            
        return image


class OCRDataset(Dataset):
    """
    Dataset for OCR fine-tuning.
    """
    def __init__(self, 
                 examples, 
                 processor, 
                 is_train=True):
        """
        Initialize dataset.
        
        Args:
            examples: Dataset examples with 'image' and 'text' fields
            processor: TrOCR processor for tokenization
            is_train: Whether this is a training dataset (for augmentation)
        """
        self.examples = examples
        self.processor = processor
        self.is_train = is_train
        self.augmentation = ImageAugmentation()
        
    def __len__(self):
        return len(self.examples)
    
    def preprocess_image(self, image):
        """
        Preprocess image for the model.
        
        Args:
            image: Input image (PIL Image or path to image)
            
        Returns:
            PIL.Image: Preprocessed image
        """
        # Load image if it's a path
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
            
        # Apply augmentation if training
        if self.is_train:
            image = self.augmentation.apply(image)
            
        # Resize to expected size
        width, height = config.IMAGE_SIZE
        image = image.resize((width, height), Image.LANCZOS)
        
        return image
    
    def __getitem__(self, idx):
        """
        Get dataset item.
        
        Args:
            idx: Index of the example
            
        Returns:
            dict: Processed example
        """
        example = self.examples[idx]
        image = example["image"]
        text = example["text"]
        
        # Preprocess image
        image = self.preprocess_image(image)
        
        # Tokenize
        encoding = self.processor(
            image, 
            text, 
            padding="max_length",
            max_length=config.MAX_LENGTH,
            truncation=True,
            return_tensors="pt"
        )
        
        # Remove batch dimension added by the processor
        for k, v in encoding.items():
            encoding[k] = v.squeeze()
            
        return encoding


def load_iam_dataset():
    """
    Load IAM dataset.
    
    Returns:
        datasets.DatasetDict: IAM dataset
    """
    iam_dataset = load_dataset(config.IAM_DATASET_NAME)
    
    # Keep only essential columns
    iam_dataset = iam_dataset.map(
        lambda example: {
            "image": example["image"],
            "text": example["text"]
        },
        remove_columns=[col for col in iam_dataset["train"].column_names 
                        if col not in ["image", "text"]]
    )
    
    # Limit number of samples for CPU training if specified
    if config.CPU_TRAINING and config.MAX_TRAIN_SAMPLES is not None:
        iam_dataset["train"] = iam_dataset["train"].select(range(min(config.MAX_TRAIN_SAMPLES, len(iam_dataset["train"]))))
    
    if "validation" in iam_dataset and config.CPU_TRAINING and config.MAX_VAL_SAMPLES is not None:
        iam_dataset["validation"] = iam_dataset["validation"].select(range(min(config.MAX_VAL_SAMPLES, len(iam_dataset["validation"]))))
    
    if "test" in iam_dataset and config.CPU_TRAINING and config.MAX_TEST_SAMPLES is not None:
        iam_dataset["test"] = iam_dataset["test"].select(range(min(config.MAX_TEST_SAMPLES, len(iam_dataset["test"]))))
    
    return iam_dataset


def load_imgur5k_dataset():
    """
    Load Imgur5K dataset if available.
    
    Returns:
        datasets.DatasetDict or None: Imgur5K dataset if available
    """
    if config.IMGUR5K_DATASET_PATH and os.path.exists(config.IMGUR5K_DATASET_PATH):
        try:
            # Load from local path
            imgur_dataset = load_dataset(
                "imagefolder", 
                data_dir=config.IMGUR5K_DATASET_PATH
            )
            
            # Limit number of samples for CPU training if specified
            if config.CPU_TRAINING and config.MAX_TRAIN_SAMPLES is not None:
                imgur_dataset["train"] = imgur_dataset["train"].select(
                    range(min(config.MAX_TRAIN_SAMPLES, len(imgur_dataset["train"])))
                )
            
            # Map to standard format if needed
            if "image" in imgur_dataset["train"].column_names and "text" in imgur_dataset["train"].column_names:
                return imgur_dataset
            else:
                # Adjust based on actual column names in the dataset
                print("Warning: Imgur5K dataset has non-standard column names. Mapping required.")
                return None
        except Exception as e:
            print(f"Error loading Imgur5K dataset: {e}")
            return None
    else:
        print("Imgur5K dataset path not provided or doesn't exist.")
        return None


def prepare_datasets():
    """
    Prepare datasets for training and evaluation.
    
    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    processor = get_processor()
    
    # Load IAM dataset
    iam_dataset = load_iam_dataset()
    
    # Load Imgur5K if available
    imgur_dataset = load_imgur5k_dataset()
    
    # Combine datasets if Imgur5K is available
    if imgur_dataset:
        # Combine train sets
        train_examples = iam_dataset["train"].concatenate(imgur_dataset["train"])
    else:
        train_examples = iam_dataset["train"]
    
    # Split train into train/validation if no validation set exists
    if "validation" not in iam_dataset:
        # Calculate split sizes
        train_size = 1.0 - config.VALIDATION_SPLIT - config.TEST_SPLIT
        val_size = config.VALIDATION_SPLIT
        test_size = config.TEST_SPLIT
        
        # Create splits
        splits = train_examples.train_test_split(
            train_size=train_size,
            test_size=val_size + test_size,
            seed=42
        )
        train_examples = splits["train"]
        
        # Further split the test portion into validation and test
        remaining_splits = splits["test"].train_test_split(
            train_size=val_size / (val_size + test_size),
            test_size=test_size / (val_size + test_size),
            seed=42
        )
        val_examples = remaining_splits["train"]
        test_examples = remaining_splits["test"]
    else:
        # Use existing splits
        val_examples = iam_dataset["validation"]
        test_examples = iam_dataset["test"]
    
    # Create datasets
    train_dataset = OCRDataset(train_examples, processor, is_train=True)
    val_dataset = OCRDataset(val_examples, processor, is_train=False)
    test_dataset = OCRDataset(test_examples, processor, is_train=False)
    
    print(f"Dataset sizes - Train: {len(train_dataset)}, Validation: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset


def get_data_loaders():
    """
    Get data loaders for training and evaluation.
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    train_dataset, val_dataset, test_dataset = prepare_datasets()
    
    # Use persistent workers only if not on CPU training
    persistent_workers = not config.CPU_TRAINING and config.NUM_WORKERS > 0
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=persistent_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=persistent_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=persistent_workers
    )
    
    return train_loader, val_loader, test_loader 