"""
Utility functions for AWS S3 integration, evaluation, and visualization
"""

import boto3
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


class S3Manager:
    """Manages S3 operations for HPE-StyleGAN2 project"""
    
    def __init__(self, bucket_name: str, region: str = 'us-east-2'):
        self.bucket_name = bucket_name
        self.region = region
        self.s3_client = boto3.client('s3', region_name=region)
        
    def upload_file(self, local_path: str, s3_key: str) -> bool:
        """Upload a file to S3"""
        try:
            self.s3_client.upload_file(local_path, self.bucket_name, s3_key)
            logger.info(f"Uploaded {local_path} to s3://{self.bucket_name}/{s3_key}")
            return True
        except Exception as e:
            logger.error(f"Failed to upload {local_path}: {e}")
            return False
    
    def download_file(self, s3_key: str, local_path: str) -> bool:
        """Download a file from S3"""
        try:
            self.s3_client.download_file(self.bucket_name, s3_key, local_path)
            logger.info(f"Downloaded s3://{self.bucket_name}/{s3_key} to {local_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to download {s3_key}: {e}")
            return False
    
    def upload_directory(self, local_dir: str, s3_prefix: str):
        """Upload a directory to S3"""
        local_path = Path(local_dir)
        
        for file_path in local_path.rglob('*'):
            if file_path.is_file():
                relative_path = file_path.relative_to(local_path)
                s3_key = f"{s3_prefix}{relative_path}"
                self.upload_file(str(file_path), s3_key)


def upload_logs_to_s3(s3_manager, logs_dir: str, s3_prefix: str):
    """Upload training logs to S3"""
    if s3_manager is None:
        return
    
    try:
        s3_manager.upload_directory(logs_dir, s3_prefix)
        logger.info(f"Logs uploaded to S3: {s3_prefix}")
    except Exception as e:
        logger.error(f"Failed to upload logs: {e}")


class HPEEvaluator:
    """Evaluates HPE-StyleGAN2 model performance"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
    
    def evaluate(self, dataloader) -> Dict[str, float]:
        """Comprehensive model evaluation"""
        self.model.eval()
        results = {}
        
        hpe_similarities = []
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(dataloader):
                real_images = batch_data['images'].to(self.device)
                hpe_targets = batch_data['hpe_embeddings'].to(self.device)
                batch_size = real_images.shape[0]
                
                # Generate fake images
                z = torch.randn(batch_size, self.model.generator.z_dim, device=self.device)
                fake_images = self.model.generator(z, hpe_targets)
                
                # Get discriminator outputs
                fake_disc_output = self.model.discriminator(fake_images, hpe_targets)
                
                # HPE similarity evaluation
                if 'hpe_embeddings' in fake_disc_output:
                    fake_hpe = fake_disc_output['hpe_embeddings']
                    hpe_sim = F.cosine_similarity(fake_hpe, hpe_targets, dim=1)
                    hpe_similarities.extend(hpe_sim.cpu().numpy())
                
                # Limit evaluation for speed
                if batch_idx >= 50:  # Evaluate on ~50 batches
                    break
        
        # Compute HPE similarity
        if hpe_similarities:
            results['hpe_similarity_mean'] = np.mean(hpe_similarities)
            results['hpe_similarity_std'] = np.std(hpe_similarities)
        
        return results


def generate_sample_grid(images: torch.Tensor, save_path: str, nrow: int = 4):
    """Generate and save a grid of sample images"""
    from torchvision.utils import make_grid, save_image
    
    # Ensure images are in correct range
    images = (images + 1) / 2  # [-1, 1] -> [0, 1]
    
    # Create grid
    grid = make_grid(images, nrow=nrow, padding=2, normalize=False)
    
    # Save image
    save_image(grid, save_path)
    logger.info(f"Sample grid saved to {save_path}")


if __name__ == "__main__":
    print("HPE-StyleGAN2 Utilities Module")
