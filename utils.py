"""
Utility functions for THINGS dataset processing and evaluation
"""
import os
import pandas as pd
import numpy as np
from PIL import Image
from typing import Tuple, List
from scipy import stats
import torch
from torchvision import transforms


def load_triplets(csv_path: str, num_samples: int = None) -> pd.DataFrame:
    """
    Load THINGS triplet data from CSV
    
    Args:
        csv_path: Path to triplet CSV file
        num_samples: Optional limit on number of triplets to load
        
    Returns:
        DataFrame with columns: image1, image2, image3, choice
    """
    print(f"Loading triplets from {csv_path}...")
    df = pd.read_csv(csv_path, sep='\t')
    
    if num_samples is not None:
        df = df.head(num_samples)
    
    print(f"Loaded {len(df)} triplets")
    return df


def load_image_paths(csv_path: str) -> List[str]:
    """
    Load image paths from THINGS image-paths.csv
    Image IDs are 1-indexed line numbers in this file
    
    Args:
        csv_path: Path to image-paths.csv
        
    Returns:
        List of image paths (index 0 = image ID 1)
    """
    with open(csv_path, 'r') as f:
        paths = [line.strip() for line in f.readlines()]
    return paths


def load_image_by_id(image_id: int, image_paths: List[str], image_root: str, transform=None) -> torch.Tensor:
    """
    Load an image by its THINGS ID (1-indexed)
    
    Args:
        image_id: 1-indexed image ID from triplet CSV
        image_paths: List of image paths from image-paths.csv
        image_root: Root directory containing extracted images
        transform: Optional torchvision transform
        
    Returns:
        Image tensor
    """
    # Convert 1-indexed ID to 0-indexed list index
    idx = image_id - 1
    
    if idx < 0 or idx >= len(image_paths):
        raise ValueError(f"Image ID {image_id} out of range [1, {len(image_paths)}]")
    
    # Get relative path and construct full path
    rel_path = image_paths[idx]
    full_path = os.path.join(image_root, rel_path)
    
    # Load image
    img = Image.open(full_path).convert('RGB')
    
    if transform is not None:
        img = transform(img)
    
    return img


def get_imagenet_transform(resize_size: int = 224):
    """
    Get standard ImageNet normalization transform for pretrained models
    """
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])


def compute_kendall_tau(predictions: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Compute accuracy and Kendall Tau correlation between predictions and ground truth
    
    Args:
        predictions: Array of predicted choices (1, 2, or 3)
        ground_truth: Array of ground truth choices (1, 2, or 3)
        
    Returns:
        Tuple of (tau, accuracy)
    """
    # Compute accuracy
    accuracy = np.mean(predictions == ground_truth)
    
    # For odd-one-out tasks, Kendall Tau can be computed as the correlation
    # between prediction agreement patterns
    # A simpler approach: compute on the predictions vs ground truth directly
    # This measures the rank correlation of choices
    try:
        tau, p_value = stats.kendalltau(predictions, ground_truth)
    except:
        # If Kendall Tau fails (e.g., all same values), just use accuracy
        tau = 0.0
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Kendall Tau: {tau:.4f}")
    
    return tau, accuracy


def triplet_to_anchor_pos_neg(triplet_row: pd.Series) -> Tuple[int, int, int]:
    """
    Convert odd-one-out triplet to (anchor, positive, negative) format
    
    Args:
        triplet_row: Row with columns image1, image2, image3, choice
        
    Returns:
        (anchor_id, positive_id, negative_id)
        
    Logic:
        - anchor = reference image (image1)
        - positive = chosen as more similar
        - negative = not chosen (the odd one out)
    """
    img1 = int(triplet_row['image1'])
    img2 = int(triplet_row['image2'])
    img3 = int(triplet_row['image3'])
    choice = int(triplet_row['choice'])
    
    # Reference is always image1
    anchor = img1
    
    # Positive is the chosen similar image
    # Negative is the odd one out
    if choice == 1:
        # Image1 was chosen as odd one out
        # This means image2 and image3 are similar
        # Use image2 as anchor instead
        anchor = img2
        positive = img3
        negative = img1
    elif choice == 2:
        # Image2 chosen as more similar to image1
        positive = img2
        negative = img3
    else:  # choice == 3
        # Image3 chosen as more similar to image1
        positive = img3
        negative = img2
    
    return anchor, positive, negative


def prepare_data_splits(triplets_df: pd.DataFrame, 
                        train_ratio: float = 0.8,
                        val_ratio: float = 0.1,
                        seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split triplets into train/val/test sets
    
    Args:
        triplets_df: Full triplet dataframe
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        seed: Random seed for reproducibility
        
    Returns:
        (train_df, val_df, test_df)
    """
    np.random.seed(seed)
    
    n = len(triplets_df)
    indices = np.random.permutation(n)
    
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    train_df = triplets_df.iloc[train_idx].reset_index(drop=True)
    val_df = triplets_df.iloc[val_idx].reset_index(drop=True)
    test_df = triplets_df.iloc[test_idx].reset_index(drop=True)
    
    print(f"Data split:")
    print(f"  Train: {len(train_df)} triplets")
    print(f"  Val: {len(val_df)} triplets")
    print(f"  Test: {len(test_df)} triplets")
    
    return train_df, val_df, test_df
