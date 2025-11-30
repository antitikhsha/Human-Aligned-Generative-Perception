#!/usr/bin/env python3
"""
Data Preparation for HPE Fine-tuning
Step 5.3: Fine-tune HPE with New Data

This script prepares new triplet similarity data for HPE fine-tuning,
including data validation, feature extraction, and AWS upload.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import boto3
from botocore.exceptions import ClientError
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureExtractor:
    """Extract features from images using pre-trained ResNet"""
    
    def __init__(self, model_name: str = 'resnet50', device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load pre-trained model
        if model_name == 'resnet50':
            self.model = models.resnet50(pretrained=True)
            self.model.fc = nn.Identity()  # Remove final classification layer
        elif model_name == 'resnet101':
            self.model = models.resnet101(pretrained=True)
            self.model.fc = nn.Identity()
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        logger.info(f"Feature extractor initialized with {model_name} on {self.device}")
    
    def extract_features(self, image_path: str) -> torch.Tensor:
        """Extract features from a single image"""
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.model(image_tensor)
            
            return features.cpu().squeeze(0)
        except Exception as e:
            logger.error(f"Failed to extract features from {image_path}: {e}")
            raise
    
    def extract_batch_features(self, image_paths: List[str], batch_size: int = 32) -> Dict[str, torch.Tensor]:
        """Extract features from multiple images in batches"""
        features_dict = {}
        
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting features"):
            batch_paths = image_paths[i:i+batch_size]
            batch_images = []
            valid_paths = []
            
            # Load and preprocess batch
            for path in batch_paths:
                try:
                    image = Image.open(path).convert('RGB')
                    image_tensor = self.transform(image)
                    batch_images.append(image_tensor)
                    valid_paths.append(path)
                except Exception as e:
                    logger.warning(f"Skipping {path}: {e}")
            
            if not batch_images:
                continue
            
            # Process batch
            batch_tensor = torch.stack(batch_images).to(self.device)
            
            with torch.no_grad():
                batch_features = self.model(batch_tensor)
            
            # Store features
            for path, features in zip(valid_paths, batch_features.cpu()):
                image_id = Path(path).stem  # Use filename without extension as ID
                features_dict[image_id] = features
        
        return features_dict

class TripletDataProcessor:
    """Process and validate triplet similarity data"""
    
    def __init__(self, s3_client=None):
        self.s3_client = s3_client or boto3.client('s3')
    
    def load_new_triplet_data(self, data_path: str) -> List[Dict]:
        """
        Load new triplet data from various formats
        Expected format: JSON with triplets containing 'anchor', 'positive', 'negative' image IDs
        """
        if data_path.endswith('.json'):
            with open(data_path, 'r') as f:
                return json.load(f)
        elif data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
            return df.to_dict('records')
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
    
    def validate_triplets(self, triplets: List[Dict], available_images: set) -> List[Dict]:
        """Validate that all images in triplets exist"""
        valid_triplets = []
        
        for i, triplet in enumerate(tqdm(triplets, desc="Validating triplets")):
            try:
                anchor = triplet['anchor']
                positive = triplet['positive']
                negative = triplet['negative']
                
                # Check if all images exist
                if all(img_id in available_images for img_id in [anchor, positive, negative]):
                    triplet['id'] = triplet.get('id', i)
                    valid_triplets.append(triplet)
                else:
                    missing = [img_id for img_id in [anchor, positive, negative] 
                              if img_id not in available_images]
                    logger.warning(f"Triplet {i}: Missing images {missing}")
            
            except KeyError as e:
                logger.warning(f"Triplet {i}: Missing required field {e}")
        
        logger.info(f"Validated {len(valid_triplets)}/{len(triplets)} triplets")
        return valid_triplets
    
    def split_triplets(self, triplets: List[Dict], train_ratio: float = 0.8) -> Tuple[List[Dict], List[Dict]]:
        """Split triplets into train and validation sets"""
        np.random.shuffle(triplets)
        split_idx = int(len(triplets) * train_ratio)
        
        train_triplets = triplets[:split_idx]
        val_triplets = triplets[split_idx:]
        
        logger.info(f"Split triplets: {len(train_triplets)} train, {len(val_triplets)} val")
        return train_triplets, val_triplets
    
    def save_triplets(self, triplets: List[Dict], output_path: str):
        """Save triplets to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(triplets, f, indent=2)
        logger.info(f"Saved {len(triplets)} triplets to {output_path}")

class AWSUploader:
    """Handle uploads to AWS S3"""
    
    def __init__(self, bucket_name: str):
        self.bucket_name = bucket_name
        self.s3_client = boto3.client('s3')
        
        # Verify bucket exists
        try:
            self.s3_client.head_bucket(Bucket=bucket_name)
            logger.info(f"Using S3 bucket: {bucket_name}")
        except ClientError:
            logger.error(f"Cannot access S3 bucket: {bucket_name}")
            raise
    
    def upload_file(self, local_path: str, s3_key: str):
        """Upload single file to S3"""
        try:
            self.s3_client.upload_file(local_path, self.bucket_name, s3_key)
            logger.info(f"Uploaded {local_path} to s3://{self.bucket_name}/{s3_key}")
        except ClientError as e:
            logger.error(f"Upload failed: {e}")
            raise
    
    def upload_features(self, features_dict: Dict[str, torch.Tensor], s3_key: str):
        """Upload features dictionary to S3"""
        # Save to temporary file
        temp_path = "/tmp/features.pt"
        torch.save(features_dict, temp_path)
        
        # Upload to S3
        self.upload_file(temp_path, s3_key)
        
        # Clean up
        os.remove(temp_path)

def prepare_hpe_data(
    image_dir: str,
    triplet_data_path: str,
    output_dir: str,
    s3_bucket: Optional[str] = None,
    s3_prefix: str = "hpe_data"
):
    """
    Main function to prepare HPE fine-tuning data
    
    Args:
        image_dir: Directory containing images
        triplet_data_path: Path to new triplet similarity data
        output_dir: Local output directory
        s3_bucket: S3 bucket for uploading (optional)
        s3_prefix: S3 prefix for uploads
    """
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize components
    feature_extractor = FeatureExtractor()
    triplet_processor = TripletDataProcessor()
    uploader = AWSUploader(s3_bucket) if s3_bucket else None
    
    logger.info("Starting HPE data preparation...")
    
    # 1. Find all images
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_paths.extend(Path(image_dir).glob(f"**/{ext}"))
    
    logger.info(f"Found {len(image_paths)} images")
    
    # 2. Extract features
    logger.info("Extracting image features...")
    features_dict = feature_extractor.extract_batch_features([str(p) for p in image_paths])
    
    # Save features locally
    features_path = output_path / "image_features.pt"
    torch.save(features_dict, features_path)
    logger.info(f"Saved features to {features_path}")
    
    # 3. Process triplet data
    logger.info("Processing triplet data...")
    triplets = triplet_processor.load_new_triplet_data(triplet_data_path)
    available_images = set(features_dict.keys())
    valid_triplets = triplet_processor.validate_triplets(triplets, available_images)
    
    # Split into train/val
    train_triplets, val_triplets = triplet_processor.split_triplets(valid_triplets)
    
    # Save triplets locally
    train_path = output_path / "train_triplets.json"
    val_path = output_path / "val_triplets.json"
    triplet_processor.save_triplets(train_triplets, str(train_path))
    triplet_processor.save_triplets(val_triplets, str(val_path))
    
    # 4. Create separate feature files for train/val
    train_image_ids = set()
    val_image_ids = set()
    
    for triplet in train_triplets:
        train_image_ids.update([triplet['anchor'], triplet['positive'], triplet['negative']])
    
    for triplet in val_triplets:
        val_image_ids.update([triplet['anchor'], triplet['positive'], triplet['negative']])
    
    # Extract train/val features
    train_features = {img_id: features_dict[img_id] for img_id in train_image_ids if img_id in features_dict}
    val_features = {img_id: features_dict[img_id] for img_id in val_image_ids if img_id in features_dict}
    
    train_features_path = output_path / "train_features.pt"
    val_features_path = output_path / "val_features.pt"
    torch.save(train_features, train_features_path)
    torch.save(val_features, val_features_path)
    
    logger.info(f"Train features: {len(train_features)} images")
    logger.info(f"Val features: {len(val_features)} images")
    
    # 5. Upload to S3 if specified
    if uploader:
        logger.info("Uploading data to S3...")
        
        files_to_upload = [
            (str(train_path), f"{s3_prefix}/triplets/train_triplets.json"),
            (str(val_path), f"{s3_prefix}/triplets/val_triplets.json"),
            (str(train_features_path), f"{s3_prefix}/features/train_features.pt"),
            (str(val_features_path), f"{s3_prefix}/features/val_features.pt"),
        ]
        
        for local_file, s3_key in files_to_upload:
            uploader.upload_file(local_file, s3_key)
    
    # 6. Generate summary report
    summary = {
        'total_images': len(image_paths),
        'processed_images': len(features_dict),
        'total_triplets': len(triplets),
        'valid_triplets': len(valid_triplets),
        'train_triplets': len(train_triplets),
        'val_triplets': len(val_triplets),
        'train_images': len(train_features),
        'val_images': len(val_features)
    }
    
    summary_path = output_path / "data_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("Data preparation completed!")
    logger.info(f"Summary: {summary}")
    
    return summary

def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(description="Prepare HPE fine-tuning data")
    parser.add_argument('--image-dir', required=True, help='Directory containing images')
    parser.add_argument('--triplets', required=True, help='Path to triplet data (JSON or CSV)')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--s3-bucket', help='S3 bucket for uploading')
    parser.add_argument('--s3-prefix', default='hpe_data', help='S3 prefix for uploads')
    
    args = parser.parse_args()
    
    try:
        summary = prepare_hpe_data(
            image_dir=args.image_dir,
            triplet_data_path=args.triplets,
            output_dir=args.output_dir,
            s3_bucket=args.s3_bucket,
            s3_prefix=args.s3_prefix
        )
        
        return 0
    except Exception as e:
        logger.error(f"Data preparation failed: {e}")
        raise

if __name__ == "__main__":
    exit(main())
