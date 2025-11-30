#!/usr/bin/env python3
"""
StyleGAN2 Training Script for AWS with THINGS Dataset Integration
Supports Human Perceptual Embeddings and distributed training
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import boto3
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

# StyleGAN2 imports (assuming you have the implementation)
try:
    from stylegan2.model import Generator, Discriminator
    from stylegan2.dataset import THINGSDataset
    from stylegan2.loss import StyleGAN2Loss
    from stylegan2.hpe import HumanPerceptualEmbeddings
except ImportError as e:
    print(f"StyleGAN2 modules not found: {e}")
    print("Please ensure StyleGAN2 implementation is available in the path")
    sys.exit(1)

class AWSTrainingConfig:
    """Configuration class for AWS training setup"""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # AWS Configuration
        self.aws_region = config.get('aws_region', 'us-east-2')
        self.data_bucket = config['data_bucket']
        self.checkpoint_bucket = config['checkpoint_bucket']
        self.output_bucket = config['output_bucket']
        
        # Training Configuration
        self.batch_size = config.get('batch_size', 8)
        self.learning_rate = config.get('learning_rate', 0.002)
        self.num_epochs = config.get('num_epochs', 100)
        self.image_size = config.get('image_size', 256)
        self.latent_dim = config.get('latent_dim', 512)
        
        # Model Configuration
        self.use_hpe = config.get('use_hpe', True)
        self.hpe_weight = config.get('hpe_weight', 1.0)
        self.gradient_penalty_weight = config.get('gradient_penalty_weight', 10.0)
        
        # Checkpointing
        self.save_interval = config.get('save_interval', 1000)
        self.checkpoint_prefix = config.get('checkpoint_prefix', 'stylegan2')
        
        # Hardware
        self.mixed_precision = config.get('mixed_precision', True)
        self.num_workers = config.get('num_workers', 4)

class S3DataManager:
    """Manages data transfer between S3 and local storage"""
    
    def __init__(self, config: AWSTrainingConfig):
        self.config = config
        self.s3_client = boto3.client('s3', region_name=config.aws_region)
        self.local_data_dir = Path('/tmp/things_data')
        self.local_data_dir.mkdir(exist_ok=True)
    
    def download_dataset(self) -> Path:
        """Download THINGS dataset from S3 to local storage"""
        logging.info("Downloading THINGS dataset from S3...")
        
        dataset_path = self.local_data_dir / 'things_dataset'
        dataset_path.mkdir(exist_ok=True)
        
        try:
            # List objects in the data bucket
            response = self.s3_client.list_objects_v2(
                Bucket=self.config.data_bucket,
                Prefix='things_dataset/'
            )
            
            if 'Contents' not in response:
                raise RuntimeError("No THINGS dataset found in S3 bucket")
            
            total_objects = len(response['Contents'])
            logging.info(f"Found {total_objects} objects to download")
            
            # Download files with progress bar
            for obj in tqdm(response['Contents'], desc="Downloading dataset"):
                local_path = dataset_path / obj['Key'].replace('things_dataset/', '')
                local_path.parent.mkdir(parents=True, exist_ok=True)
                
                self.s3_client.download_file(
                    self.config.data_bucket,
                    obj['Key'],
                    str(local_path)
                )
            
            logging.info(f"Dataset downloaded to {dataset_path}")
            return dataset_path
            
        except Exception as e:
            logging.error(f"Failed to download dataset: {e}")
            raise
    
    def upload_checkpoint(self, checkpoint_path: Path, epoch: int, step: int):
        """Upload model checkpoint to S3"""
        s3_key = f"checkpoints/{self.config.checkpoint_prefix}_epoch_{epoch}_step_{step}.pt"
        
        try:
            self.s3_client.upload_file(
                str(checkpoint_path),
                self.config.checkpoint_bucket,
                s3_key
            )
            logging.info(f"Checkpoint uploaded to s3://{self.config.checkpoint_bucket}/{s3_key}")
            
        except Exception as e:
            logging.error(f"Failed to upload checkpoint: {e}")
            raise
    
    def upload_outputs(self, output_dir: Path):
        """Upload training outputs to S3"""
        for file_path in output_dir.rglob('*'):
            if file_path.is_file():
                s3_key = f"outputs/{file_path.relative_to(output_dir)}"
                
                try:
                    self.s3_client.upload_file(
                        str(file_path),
                        self.config.output_bucket,
                        s3_key
                    )
                except Exception as e:
                    logging.warning(f"Failed to upload {file_path}: {e}")

class StyleGAN2Trainer:
    """Main trainer class for StyleGAN2 with HPE integration"""
    
    def __init__(self, config: AWSTrainingConfig, rank: int = 0, world_size: int = 1):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
        
        # Initialize models
        self.generator = Generator(
            latent_dim=config.latent_dim,
            img_resolution=config.image_size
        ).to(self.device)
        
        self.discriminator = Discriminator(
            img_resolution=config.image_size
        ).to(self.device)
        
        # Initialize HPE if enabled
        if config.use_hpe:
            self.hpe = HumanPerceptualEmbeddings().to(self.device)
        else:
            self.hpe = None
        
        # Loss function
        self.criterion = StyleGAN2Loss(
            hpe_model=self.hpe,
            hpe_weight=config.hpe_weight,
            gp_weight=config.gradient_penalty_weight
        )
        
        # Optimizers
        self.optimizer_g = torch.optim.Adam(
            self.generator.parameters(),
            lr=config.learning_rate,
            betas=(0.0, 0.99)
        )
        
        self.optimizer_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=config.learning_rate,
            betas=(0.0, 0.99)
        )
        
        # Mixed precision training
        if config.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        
        # Distributed training setup
        if world_size > 1:
            self.generator = torch.nn.parallel.DistributedDataParallel(
                self.generator, device_ids=[rank]
            )
            self.discriminator = torch.nn.parallel.DistributedDataParallel(
                self.discriminator, device_ids=[rank]
            )
        
        # Logging
        if rank == 0:
            self.writer = SummaryWriter(log_dir='runs/stylegan2_training')
        else:
            self.writer = None
        
        # S3 manager
        self.s3_manager = S3DataManager(config)
    
    def prepare_dataset(self) -> DataLoader:
        """Prepare THINGS dataset with data loaders"""
        # Download dataset from S3
        dataset_path = self.s3_manager.download_dataset()
        
        # Create dataset
        transform = transforms.Compose([
            transforms.Resize((self.config.image_size, self.config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        dataset = THINGSDataset(
            root_dir=dataset_path,
            transform=transform,
            load_hpe=self.config.use_hpe
        )
        
        # Create data loader
        if self.world_size > 1:
            sampler = DistributedSampler(dataset, rank=self.rank)
        else:
            sampler = None
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        return dataloader
    
    def train_step(self, real_images: torch.Tensor, hpe_embeddings: Optional[torch.Tensor] = None):
        """Single training step"""
        batch_size = real_images.size(0)
        
        # Generate noise
        noise = torch.randn(batch_size, self.config.latent_dim, device=self.device)
        
        # Train Discriminator
        self.optimizer_d.zero_grad()
        
        if self.scaler:
            with torch.cuda.amp.autocast():
                fake_images = self.generator(noise)
                d_loss = self.criterion.discriminator_loss(
                    real_images, fake_images, self.discriminator
                )
            self.scaler.scale(d_loss).backward()
            self.scaler.step(self.optimizer_d)
            self.scaler.update()
        else:
            fake_images = self.generator(noise)
            d_loss = self.criterion.discriminator_loss(
                real_images, fake_images, self.discriminator
            )
            d_loss.backward()
            self.optimizer_d.step()
        
        # Train Generator
        self.optimizer_g.zero_grad()
        
        if self.scaler:
            with torch.cuda.amp.autocast():
                noise = torch.randn(batch_size, self.config.latent_dim, device=self.device)
                fake_images = self.generator(noise)
                g_loss = self.criterion.generator_loss(
                    fake_images, self.discriminator, hpe_embeddings
                )
            self.scaler.scale(g_loss).backward()
            self.scaler.step(self.optimizer_g)
            self.scaler.update()
        else:
            noise = torch.randn(batch_size, self.config.latent_dim, device=self.device)
            fake_images = self.generator(noise)
            g_loss = self.criterion.generator_loss(
                fake_images, self.discriminator, hpe_embeddings
            )
            g_loss.backward()
            self.optimizer_g.step()
        
        return {
            'g_loss': g_loss.item(),
            'd_loss': d_loss.item(),
            'fake_images': fake_images.detach()
        }
    
    def save_checkpoint(self, epoch: int, step: int):
        """Save model checkpoint"""
        if self.rank != 0:
            return
        
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'optimizer_d_state_dict': self.optimizer_d.state_dict(),
            'config': vars(self.config)
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        checkpoint_path = Path(f'checkpoint_epoch_{epoch}_step_{step}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # Upload to S3
        self.s3_manager.upload_checkpoint(checkpoint_path, epoch, step)
        
        # Clean up local checkpoint
        checkpoint_path.unlink()
    
    def train(self):
        """Main training loop"""
        dataloader = self.prepare_dataset()
        
        global_step = 0
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            if self.rank == 0:
                logging.info(f"Starting epoch {epoch + 1}/{self.config.num_epochs}")
            
            # Set epoch for distributed sampler
            if hasattr(dataloader.sampler, 'set_epoch'):
                dataloader.sampler.set_epoch(epoch)
            
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            
            for batch_idx, batch in enumerate(tqdm(dataloader, disable=(self.rank != 0))):
                if isinstance(batch, dict):
                    real_images = batch['image'].to(self.device)
                    hpe_embeddings = batch.get('hpe_embedding')
                    if hpe_embeddings is not None:
                        hpe_embeddings = hpe_embeddings.to(self.device)
                else:
                    real_images = batch.to(self.device)
                    hpe_embeddings = None
                
                # Training step
                losses = self.train_step(real_images, hpe_embeddings)
                
                epoch_g_loss += losses['g_loss']
                epoch_d_loss += losses['d_loss']
                global_step += 1
                
                # Logging
                if self.rank == 0 and global_step % 100 == 0:
                    self.writer.add_scalar('Loss/Generator', losses['g_loss'], global_step)
                    self.writer.add_scalar('Loss/Discriminator', losses['d_loss'], global_step)
                    
                    # Log sample images
                    if global_step % 1000 == 0:
                        fake_images = losses['fake_images'][:8]
                        self.writer.add_images(
                            'Generated_Images',
                            (fake_images + 1) / 2,
                            global_step
                        )
                
                # Save checkpoint
                if global_step % self.config.save_interval == 0:
                    self.save_checkpoint(epoch, global_step)
            
            # Log epoch statistics
            if self.rank == 0:
                avg_g_loss = epoch_g_loss / len(dataloader)
                avg_d_loss = epoch_d_loss / len(dataloader)
                
                elapsed_time = time.time() - start_time
                logging.info(
                    f"Epoch {epoch + 1} completed in {elapsed_time:.2f}s - "
                    f"G Loss: {avg_g_loss:.4f}, D Loss: {avg_d_loss:.4f}"
                )
                
                self.writer.add_scalar('Epoch/Generator_Loss', avg_g_loss, epoch)
                self.writer.add_scalar('Epoch/Discriminator_Loss', avg_d_loss, epoch)
        
        # Upload final outputs
        if self.rank == 0:
            self.s3_manager.upload_outputs(Path('runs'))
            logging.info("Training completed and outputs uploaded to S3")

def setup_distributed(rank: int, world_size: int):
    """Setup distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    dist.init_process_group(
        backend='nccl',
        rank=rank,
        world_size=world_size
    )

def cleanup_distributed():
    """Cleanup distributed training"""
    dist.destroy_process_group()

def train_worker(rank: int, world_size: int, config_path: str):
    """Worker function for distributed training"""
    if world_size > 1:
        setup_distributed(rank, world_size)
    
    try:
        config = AWSTrainingConfig(config_path)
        trainer = StyleGAN2Trainer(config, rank, world_size)
        trainer.train()
    finally:
        if world_size > 1:
            cleanup_distributed()

def main():
    parser = argparse.ArgumentParser(description='StyleGAN2 Training on AWS')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--distributed', action='store_true', help='Enable distributed training')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Check GPU availability
    if not torch.cuda.is_available():
        logging.error("CUDA not available. This training requires GPU.")
        return
    
    world_size = torch.cuda.device_count() if args.distributed else 1
    
    logging.info(f"Starting training with {world_size} GPU(s)")
    
    if world_size > 1:
        mp.spawn(
            train_worker,
            args=(world_size, args.config),
            nprocs=world_size,
            join=True
        )
    else:
        train_worker(0, 1, args.config)

if __name__ == '__main__':
    main()
