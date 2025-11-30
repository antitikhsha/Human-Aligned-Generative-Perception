#!/usr/bin/env python3
"""
Step 5.3: Fine-tune HPE with New Data on AWS
Human-Aligned Generative Perception Research Project

This script fine-tunes existing Human Perceptual Embeddings (HPE) with new triplet
similarity data, integrating with AWS infrastructure for scalable training.
"""

import os
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import boto3
from botocore.exceptions import ClientError
import yaml
from tqdm import tqdm
import wandb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HPEModel(nn.Module):
    """
    Human Perceptual Embeddings model for learning similarity representations
    from triplet judgments (odd-one-out tasks).
    """
    
    def __init__(self, input_dim: int = 2048, embedding_dim: int = 66, dropout: float = 0.1):
        super(HPEModel, self).__init__()
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        
        # Feature encoder network
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(256, embedding_dim)
        )
        
        # L2 normalization for embeddings
        self.normalize = nn.functional.normalize
        
    def forward(self, x):
        """Forward pass through the HPE model"""
        embeddings = self.encoder(x)
        # L2 normalize embeddings for cosine similarity
        return self.normalize(embeddings, p=2, dim=1)
    
    def compute_triplet_loss(self, anchor, positive, negative, margin: float = 1.0):
        """
        Compute triplet loss for odd-one-out tasks
        Args:
            anchor: Reference image embedding
            positive: Similar image embedding (should be closer to anchor)
            negative: Dissimilar image embedding (should be farther from anchor)
            margin: Margin for triplet loss
        """
        pos_distance = torch.norm(anchor - positive, p=2, dim=1)
        neg_distance = torch.norm(anchor - negative, p=2, dim=1)
        
        loss = torch.clamp(pos_distance - neg_distance + margin, min=0.0)
        return torch.mean(loss)

class TripletDataset(Dataset):
    """Dataset for loading triplet similarity judgments"""
    
    def __init__(self, triplet_data_path: str, features_path: str, transform=None):
        """
        Args:
            triplet_data_path: Path to triplet judgments JSON file
            features_path: Path to pre-extracted image features (e.g., ResNet features)
            transform: Optional data transforms
        """
        self.transform = transform
        
        # Load triplet judgments
        with open(triplet_data_path, 'r') as f:
            self.triplets = json.load(f)
        
        # Load pre-extracted features
        self.features = torch.load(features_path)
        
        logger.info(f"Loaded {len(self.triplets)} triplets with features for {len(self.features)} images")
    
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        triplet = self.triplets[idx]
        
        # Get image IDs from triplet judgment
        anchor_id = triplet['anchor']
        positive_id = triplet['positive']  # More similar to anchor
        negative_id = triplet['negative']  # Less similar to anchor
        
        # Extract corresponding features
        anchor_feat = self.features[anchor_id]
        positive_feat = self.features[positive_id]
        negative_feat = self.features[negative_id]
        
        if self.transform:
            anchor_feat = self.transform(anchor_feat)
            positive_feat = self.transform(positive_feat)
            negative_feat = self.transform(negative_feat)
        
        return {
            'anchor': anchor_feat,
            'positive': positive_feat,
            'negative': negative_feat,
            'triplet_id': triplet.get('id', idx)
        }

class HPEFineTuner:
    """Main class for fine-tuning HPE models with new data"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize AWS clients
        self.s3_client = boto3.client('s3')
        self.ec2_client = boto3.client('ec2')
        
        # Setup model
        self.model = None
        self.optimizer = None
        self.scheduler = None
        
        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')
        
    def setup_model(self):
        """Initialize or load existing HPE model"""
        model_config = self.config['model']
        
        self.model = HPEModel(
            input_dim=model_config['input_dim'],
            embedding_dim=model_config['embedding_dim'],
            dropout=model_config.get('dropout', 0.1)
        ).to(self.device)
        
        # Load pre-trained weights if available
        if self.config.get('pretrained_model_path'):
            self._load_pretrained_model()
        
        # Setup optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training'].get('weight_decay', 1e-5)
        )
        
        # Setup learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        logger.info(f"Model initialized with {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def _load_pretrained_model(self):
        """Load pre-trained HPE model from S3 or local path"""
        model_path = self.config['pretrained_model_path']
        
        if model_path.startswith('s3://'):
            # Download from S3
            bucket, key = self._parse_s3_path(model_path)
            local_path = f"/tmp/{Path(key).name}"
            self.s3_client.download_file(bucket, key, local_path)
            model_path = local_path
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            if 'optimizer_state_dict' in checkpoint and self.config.get('resume_training', False):
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.current_epoch = checkpoint.get('epoch', 0)
                self.best_loss = checkpoint.get('best_loss', float('inf'))
            
            logger.info(f"Loaded pre-trained model from {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load pre-trained model: {e}")
            raise
    
    def prepare_datasets(self):
        """Load and prepare training and validation datasets"""
        data_config = self.config['data']
        
        # Download data from S3 if needed
        train_triplets = self._download_if_s3(data_config['train_triplets_path'])
        val_triplets = self._download_if_s3(data_config['val_triplets_path'])
        train_features = self._download_if_s3(data_config['train_features_path'])
        val_features = self._download_if_s3(data_config['val_features_path'])
        
        # Create datasets
        self.train_dataset = TripletDataset(train_triplets, train_features)
        self.val_dataset = TripletDataset(val_triplets, val_features)
        
        # Create data loaders
        train_config = self.config['training']
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=train_config['batch_size'],
            shuffle=True,
            num_workers=train_config.get('num_workers', 4),
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=train_config['batch_size'],
            shuffle=False,
            num_workers=train_config.get('num_workers', 4),
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        logger.info(f"Prepared datasets: {len(self.train_dataset)} train, {len(self.val_dataset)} val samples")
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        
        for batch in progress_bar:
            # Move batch to device
            anchor = batch['anchor'].to(self.device)
            positive = batch['positive'].to(self.device)
            negative = batch['negative'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            anchor_emb = self.model(anchor)
            positive_emb = self.model(positive)
            negative_emb = self.model(negative)
            
            # Compute triplet loss
            loss = self.model.compute_triplet_loss(
                anchor_emb, positive_emb, negative_emb,
                margin=self.config['training'].get('triplet_margin', 1.0)
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config['training'].get('grad_clip_norm'):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['grad_clip_norm']
                )
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            avg_loss = total_loss / num_batches
            progress_bar.set_postfix({'Loss': f'{avg_loss:.4f}'})
            
            # Log to wandb if enabled
            if self.config.get('use_wandb', False):
                wandb.log({
                    'train_loss_batch': loss.item(),
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
        
        return total_loss / num_batches
    
    def validate(self) -> float:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                # Move batch to device
                anchor = batch['anchor'].to(self.device)
                positive = batch['positive'].to(self.device)
                negative = batch['negative'].to(self.device)
                
                # Forward pass
                anchor_emb = self.model(anchor)
                positive_emb = self.model(positive)
                negative_emb = self.model(negative)
                
                # Compute loss
                loss = self.model.compute_triplet_loss(
                    anchor_emb, positive_emb, negative_emb,
                    margin=self.config['training'].get('triplet_margin', 1.0)
                )
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def save_checkpoint(self, epoch: int, loss: float, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'best_loss': self.best_loss,
            'config': self.config
        }
        
        # Save locally first
        checkpoint_dir = Path(self.config['output']['checkpoint_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"hpe_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = checkpoint_dir / "hpe_best.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model with loss {loss:.4f}")
        
        # Upload to S3 if configured
        if self.config['output'].get('s3_bucket'):
            self._upload_checkpoint_to_s3(checkpoint_path, epoch, is_best)
        
        logger.info(f"Saved checkpoint for epoch {epoch}")
    
    def _upload_checkpoint_to_s3(self, local_path: Path, epoch: int, is_best: bool):
        """Upload checkpoint to S3"""
        bucket = self.config['output']['s3_bucket']
        s3_prefix = self.config['output'].get('s3_prefix', 'hpe_checkpoints')
        
        # Upload regular checkpoint
        s3_key = f"{s3_prefix}/hpe_epoch_{epoch}.pt"
        self._upload_to_s3(local_path, bucket, s3_key)
        
        # Upload best model
        if is_best:
            s3_key_best = f"{s3_prefix}/hpe_best.pt"
            self._upload_to_s3(local_path, bucket, s3_key_best)
    
    def _upload_to_s3(self, local_path: Path, bucket: str, key: str):
        """Upload file to S3"""
        try:
            self.s3_client.upload_file(str(local_path), bucket, key)
            logger.info(f"Uploaded {local_path} to s3://{bucket}/{key}")
        except ClientError as e:
            logger.error(f"Failed to upload to S3: {e}")
    
    def _download_if_s3(self, path: str) -> str:
        """Download file from S3 if path starts with s3://, otherwise return original path"""
        if not path.startswith('s3://'):
            return path
        
        bucket, key = self._parse_s3_path(path)
        local_path = f"/tmp/{Path(key).name}"
        
        try:
            self.s3_client.download_file(bucket, key, local_path)
            logger.info(f"Downloaded {path} to {local_path}")
            return local_path
        except ClientError as e:
            logger.error(f"Failed to download from S3: {e}")
            raise
    
    def _parse_s3_path(self, s3_path: str) -> Tuple[str, str]:
        """Parse S3 path into bucket and key"""
        path_parts = s3_path.replace('s3://', '').split('/', 1)
        bucket = path_parts[0]
        key = path_parts[1] if len(path_parts) > 1 else ''
        return bucket, key
    
    def train(self):
        """Main training loop"""
        train_config = self.config['training']
        num_epochs = train_config['num_epochs']
        
        # Initialize wandb if enabled
        if self.config.get('use_wandb', False):
            wandb.init(
                project=self.config.get('wandb_project', 'hpe-finetuning'),
                config=self.config,
                name=f"hpe_finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
        
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(self.current_epoch, num_epochs):
            # Training
            train_loss = self.train_epoch(epoch)
            
            # Validation
            val_loss = self.validate()
            
            # Update learning rate scheduler
            self.scheduler.step(val_loss)
            
            # Log metrics
            logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if self.config.get('use_wandb', False):
                wandb.log({
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
            
            # Save checkpoint
            is_best = val_loss < self.best_loss
            if is_best:
                self.best_loss = val_loss
            
            # Save checkpoint every few epochs or if best
            if (epoch + 1) % train_config.get('save_freq', 5) == 0 or is_best:
                self.save_checkpoint(epoch + 1, val_loss, is_best)
        
        logger.info("Training completed!")
        
        if self.config.get('use_wandb', False):
            wandb.finish()

def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_aws_environment():
    """Setup AWS environment and verify credentials"""
    try:
        # Verify AWS credentials
        sts_client = boto3.client('sts')
        identity = sts_client.get_caller_identity()
        logger.info(f"AWS Identity: {identity['Arn']}")
        
        # Check if running on EC2
        try:
            ec2_metadata = boto3.Session().region_name
            logger.info(f"Running on EC2 in region: {ec2_metadata}")
        except:
            logger.info("Running outside of EC2")
        
        return True
    except Exception as e:
        logger.error(f"AWS setup failed: {e}")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Fine-tune HPE with new triplet data")
    parser.add_argument('--config', required=True, help='Path to configuration YAML file')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    config = load_config(args.config)
    
    if args.resume:
        config['resume_training'] = True
    
    # Setup AWS environment
    if not setup_aws_environment():
        logger.error("Failed to setup AWS environment")
        return 1
    
    try:
        # Initialize fine-tuner
        fine_tuner = HPEFineTuner(config)
        
        # Setup model and data
        fine_tuner.setup_model()
        fine_tuner.prepare_datasets()
        
        # Start training
        fine_tuner.train()
        
        logger.info("HPE fine-tuning completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    exit(main())
