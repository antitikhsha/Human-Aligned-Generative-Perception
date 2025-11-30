"""
HPE-StyleGAN2 Training Loop
Implements training with multiple loss components:
1. Adversarial loss (GAN)
2. HPE alignment loss
3. Perceptual loss
4. R1 regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
import wandb
from pathlib import Path
import time
from collections import defaultdict

from ..stylegan2.hpe_stylegan2 import HPEStyleGAN2, Generator, HPEDiscriminator
from ..hpe.hpe_core import TripletLoss, compute_hpe_similarity

logger = logging.getLogger(__name__)


class HPEStyleGAN2Loss:
    """
    Multi-objective loss function for HPE-StyleGAN2 training
    """
    
    def __init__(
        self,
        device: torch.device,
        lambda_hpe: float = 1.0,
        lambda_perceptual: float = 0.1,
        lambda_r1: float = 10.0,
        lambda_triplet: float = 0.5
    ):
        self.device = device
        self.lambda_hpe = lambda_hpe
        self.lambda_perceptual = lambda_perceptual
        self.lambda_r1 = lambda_r1
        self.lambda_triplet = lambda_triplet
        
        # Loss functions
        self.triplet_loss = TripletLoss(margin=0.3)
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        
        # Initialize VGG for perceptual loss
        self._init_perceptual_network()
    
    def _init_perceptual_network(self):
        """Initialize VGG network for perceptual loss"""
        try:
            import torchvision.models as models
            vgg = models.vgg19(pretrained=True).features
            
            # Extract specific layers for perceptual loss
            self.perceptual_layers = nn.ModuleList([
                vgg[:4],   # conv1_2
                vgg[:9],   # conv2_2
                vgg[:18],  # conv3_4
                vgg[:27],  # conv4_4
                vgg[:36]   # conv5_4
            ]).to(self.device)
            
            # Freeze VGG parameters
            for layer in self.perceptual_layers:
                for param in layer.parameters():
                    param.requires_grad = False
                    
        except ImportError:
            logger.warning("torchvision not available, skipping perceptual loss")
            self.perceptual_layers = None
    
    def perceptual_loss(self, fake_images: torch.Tensor, real_images: torch.Tensor) -> torch.Tensor:
        """Compute perceptual loss using VGG features"""
        if self.perceptual_layers is None:
            return torch.tensor(0.0, device=self.device)
        
        loss = 0.0
        fake_x = fake_images
        real_x = real_images
        
        for layer in self.perceptual_layers:
            fake_x = layer(fake_x)
            real_x = layer(real_x)
            loss += F.mse_loss(fake_x, real_x)
        
        return loss / len(self.perceptual_layers)
    
    def r1_regularization(
        self, 
        real_images: torch.Tensor, 
        discriminator: nn.Module
    ) -> torch.Tensor:
        """Compute R1 gradient penalty"""
        real_images.requires_grad_(True)
        
        disc_output = discriminator(real_images)
        real_scores = disc_output['adv_logits']
        
        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=real_scores.sum(),
            inputs=real_images,
            create_graph=True,
            retain_graph=True
        )[0]
        
        # R1 penalty
        penalty = gradients.pow(2).sum([1, 2, 3]).mean()
        
        return penalty
    
    def compute_generator_loss(
        self,
        fake_images: torch.Tensor,
        fake_disc_output: Dict[str, torch.Tensor],
        real_images: torch.Tensor,
        hpe_targets: torch.Tensor,
        triplet_data: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute generator losses"""
        losses = {}
        total_loss = 0.0
        
        # Adversarial loss
        adv_loss = -fake_disc_output['adv_logits'].mean()
        losses['g_adv'] = adv_loss
        total_loss += adv_loss
        
        # HPE adversarial loss
        if 'hpe_combined_logits' in fake_disc_output:
            hpe_adv_loss = -fake_disc_output['hpe_combined_logits'].mean()
            losses['g_hpe_adv'] = hpe_adv_loss
            total_loss += self.lambda_hpe * hpe_adv_loss
        
        # HPE alignment loss
        if 'hpe_embeddings' in fake_disc_output:
            fake_hpe = fake_disc_output['hpe_embeddings']
            hpe_align_loss = self.mse_loss(fake_hpe, hpe_targets)
            losses['g_hpe_align'] = hpe_align_loss
            total_loss += self.lambda_hpe * hpe_align_loss
        
        # Perceptual loss
        perceptual = self.perceptual_loss(fake_images, real_images)
        losses['g_perceptual'] = perceptual
        total_loss += self.lambda_perceptual * perceptual
        
        # Triplet loss for HPE consistency
        if triplet_data is not None and 'hpe_embeddings' in fake_disc_output:
            anchor_hpe, pos_hpe, neg_hpe = triplet_data
            fake_hpe = fake_disc_output['hpe_embeddings']
            
            # Ensure fake_hpe matches triplet batch size
            if fake_hpe.shape[0] == anchor_hpe.shape[0]:
                triplet = self.triplet_loss(fake_hpe, pos_hpe, neg_hpe)
                losses['g_triplet'] = triplet
                total_loss += self.lambda_triplet * triplet
        
        losses['g_total'] = total_loss
        return losses
    
    def compute_discriminator_loss(
        self,
        real_images: torch.Tensor,
        fake_images: torch.Tensor,
        real_disc_output: Dict[str, torch.Tensor],
        fake_disc_output: Dict[str, torch.Tensor],
        hpe_targets: torch.Tensor,
        apply_r1: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Compute discriminator losses"""
        losses = {}
        total_loss = 0.0
        
        # Standard adversarial loss
        real_loss = F.softplus(-real_disc_output['adv_logits']).mean()
        fake_loss = F.softplus(fake_disc_output['adv_logits']).mean()
        adv_loss = real_loss + fake_loss
        
        losses['d_real'] = real_loss
        losses['d_fake'] = fake_loss
        losses['d_adv'] = adv_loss
        total_loss += adv_loss
        
        # HPE-conditioned adversarial loss
        if 'hpe_combined_logits' in real_disc_output:
            real_hpe_loss = F.softplus(-real_disc_output['hpe_combined_logits']).mean()
            fake_hpe_loss = F.softplus(fake_disc_output['hpe_combined_logits']).mean()
            hpe_adv_loss = real_hpe_loss + fake_hpe_loss
            
            losses['d_hpe_real'] = real_hpe_loss
            losses['d_hpe_fake'] = fake_hpe_loss
            losses['d_hpe_adv'] = hpe_adv_loss
            total_loss += self.lambda_hpe * hpe_adv_loss
        
        # HPE alignment loss on real images
        if 'hpe_embeddings' in real_disc_output:
            real_hpe = real_disc_output['hpe_embeddings']
            hpe_real_loss = self.mse_loss(real_hpe, hpe_targets)
            losses['d_hpe_real_align'] = hpe_real_loss
            total_loss += self.lambda_hpe * hpe_real_loss
        
        # R1 regularization
        if apply_r1:
            r1_penalty = self.r1_regularization(real_images, lambda x: real_disc_output)
            losses['d_r1'] = r1_penalty
            total_loss += self.lambda_r1 * r1_penalty
        
        losses['d_total'] = total_loss
        return losses


class HPEStyleGAN2Trainer:
    """
    Trainer for HPE-StyleGAN2 model
    """
    
    def __init__(
        self,
        model: HPEStyleGAN2,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        learning_rates: Dict[str, float] = None,
        loss_config: Dict[str, float] = None,
        checkpoint_dir: str = "./checkpoints",
        log_dir: str = "./logs"
    ):
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        
        # Default learning rates
        if learning_rates is None:
            learning_rates = {'g_lr': 0.0002, 'd_lr': 0.0002}
        
        # Default loss configuration
        if loss_config is None:
            loss_config = {
                'lambda_hpe': 1.0,
                'lambda_perceptual': 0.1,
                'lambda_r1': 10.0,
                'lambda_triplet': 0.5
            }
        
        # Initialize loss function
        self.loss_fn = HPEStyleGAN2Loss(device=device, **loss_config)
        
        # Initialize optimizers
        self.g_optimizer = optim.Adam(
            self.model.generator.parameters(),
            lr=learning_rates['g_lr'],
            betas=(0.0, 0.99)
        )
        
        self.d_optimizer = optim.Adam(
            self.model.discriminator.parameters(),
            lr=learning_rates['d_lr'],
            betas=(0.0, 0.99)
        )
        
        # Directories
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        
        # Metrics tracking
        self.train_metrics = defaultdict(list)
        self.val_metrics = defaultdict(list)
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train one epoch"""
        self.model.train()
        epoch_metrics = defaultdict(list)
        
        for batch_idx, batch_data in enumerate(self.train_dataloader):
            # Parse batch data
            real_images = batch_data['images'].to(self.device)
            hpe_targets = batch_data['hpe_embeddings'].to(self.device)
            batch_size = real_images.shape[0]
            
            # Generate random latents
            z = torch.randn(batch_size, self.model.generator.z_dim, device=self.device)
            
            # Generate fake images
            fake_images = self.model.generator(z, hpe_targets)
            
            # === Train Discriminator ===
            self.d_optimizer.zero_grad()
            
            # Get discriminator outputs
            real_disc_output = self.model.discriminator(real_images, hpe_targets)
            fake_disc_output = self.model.discriminator(fake_images.detach(), hpe_targets)
            
            # Compute discriminator loss
            apply_r1 = (self.global_step % 16 == 0)  # R1 every 16 steps
            d_losses = self.loss_fn.compute_discriminator_loss(
                real_images=real_images,
                fake_images=fake_images.detach(),
                real_disc_output=real_disc_output,
                fake_disc_output=fake_disc_output,
                hpe_targets=hpe_targets,
                apply_r1=apply_r1
            )
            
            d_losses['d_total'].backward()
            self.d_optimizer.step()
            
            # === Train Generator ===
            self.g_optimizer.zero_grad()
            
            # Get fresh discriminator outputs for generator
            fake_disc_output = self.model.discriminator(fake_images, hpe_targets)
            
            # Get triplet data if available
            triplet_data = None
            if 'triplet_anchor' in batch_data:
                anchor_hpe = batch_data['triplet_anchor'].to(self.device)
                pos_hpe = batch_data['triplet_positive'].to(self.device)
                neg_hpe = batch_data['triplet_negative'].to(self.device)
                triplet_data = (anchor_hpe, pos_hpe, neg_hpe)
            
            # Compute generator loss
            g_losses = self.loss_fn.compute_generator_loss(
                fake_images=fake_images,
                fake_disc_output=fake_disc_output,
                real_images=real_images,
                hpe_targets=hpe_targets,
                triplet_data=triplet_data
            )
            
            g_losses['g_total'].backward()
            self.g_optimizer.step()
            
            # === Logging ===
            # Combine losses
            all_losses = {**d_losses, **g_losses}
            
            for key, value in all_losses.items():
                epoch_metrics[key].append(value.item())
            
            self.global_step += 1
            
            # Log every 100 steps
            if batch_idx % 100 == 0:
                logger.info(
                    f"Epoch {epoch}, Batch {batch_idx}/{len(self.train_dataloader)}, "
                    f"G_loss: {g_losses['g_total'].item():.4f}, "
                    f"D_loss: {d_losses['d_total'].item():.4f}"
                )
                
                # Log to wandb if available
                if wandb.run is not None:
                    wandb.log({
                        'step': self.global_step,
                        'epoch': epoch,
                        **{k: v.item() for k, v in all_losses.items()}
                    })
        
        # Average metrics for epoch
        epoch_avg = {key: np.mean(values) for key, values in epoch_metrics.items()}
        return epoch_avg
    
    def validate(self) -> Dict[str, float]:
        """Validate the model"""
        if self.val_dataloader is None:
            return {}
        
        self.model.eval()
        val_metrics = defaultdict(list)
        
        with torch.no_grad():
            for batch_data in self.val_dataloader:
                real_images = batch_data['images'].to(self.device)
                hpe_targets = batch_data['hpe_embeddings'].to(self.device)
                batch_size = real_images.shape[0]
                
                # Generate fake images
                z = torch.randn(batch_size, self.model.generator.z_dim, device=self.device)
                fake_images = self.model.generator(z, hpe_targets)
                
                # Compute discriminator outputs
                real_disc_output = self.model.discriminator(real_images, hpe_targets)
                fake_disc_output = self.model.discriminator(fake_images, hpe_targets)
                
                # Compute losses
                d_losses = self.loss_fn.compute_discriminator_loss(
                    real_images=real_images,
                    fake_images=fake_images,
                    real_disc_output=real_disc_output,
                    fake_disc_output=fake_disc_output,
                    hpe_targets=hpe_targets,
                    apply_r1=False
                )
                
                g_losses = self.loss_fn.compute_generator_loss(
                    fake_images=fake_images,
                    fake_disc_output=fake_disc_output,
                    real_images=real_images,
                    hpe_targets=hpe_targets
                )
                
                # Store metrics
                all_losses = {**d_losses, **g_losses}
                for key, value in all_losses.items():
                    val_metrics[f'val_{key}'].append(value.item())
        
        # Average validation metrics
        val_avg = {key: np.mean(values) for key, values in val_metrics.items()}
        return val_avg
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'generator_state_dict': self.model.generator.state_dict(),
            'discriminator_state_dict': self.model.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'metrics': metrics,
            'model_config': {
                'z_dim': self.model.generator.z_dim,
                'w_dim': self.model.generator.w_dim,
                'img_resolution': self.model.generator.img_resolution,
                'img_channels': self.model.generator.img_channels
            }
        }
        
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if 'val_g_total' in metrics and metrics['val_g_total'] < min(self.val_metrics.get('val_g_total', [float('inf')])):
            best_path = self.checkpoint_dir / 'best_checkpoint.pt'
            torch.save(checkpoint, best_path)
            logger.info(f"New best checkpoint saved: {best_path}")
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.model.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
        return checkpoint['metrics']
    
    def train(
        self, 
        num_epochs: int, 
        save_every: int = 10,
        validate_every: int = 5
    ):
        """Complete training loop"""
        logger.info(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(self.current_epoch, self.current_epoch + num_epochs):
            start_time = time.time()
            
            # Train
            train_metrics = self.train_epoch(epoch)
            self.train_metrics.update(train_metrics)
            
            # Validate
            val_metrics = {}
            if epoch % validate_every == 0:
                val_metrics = self.validate()
                self.val_metrics.update(val_metrics)
            
            # Combine metrics
            all_metrics = {**train_metrics, **val_metrics}
            
            # Log epoch summary
            epoch_time = time.time() - start_time
            logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s")
            logger.info(f"Train G_loss: {train_metrics.get('g_total', 0):.4f}, D_loss: {train_metrics.get('d_total', 0):.4f}")
            
            if val_metrics:
                logger.info(f"Val G_loss: {val_metrics.get('val_g_total', 0):.4f}, D_loss: {val_metrics.get('val_d_total', 0):.4f}")
            
            # Save checkpoint
            if epoch % save_every == 0:
                self.save_checkpoint(epoch, all_metrics)
            
            self.current_epoch = epoch + 1
        
        logger.info("Training completed!")


if __name__ == "__main__":
    # Example training setup
    print("HPE-StyleGAN2 Training Module")
    
    # This would be used with actual data loaders
    # trainer = HPEStyleGAN2Trainer(
    #     model=hpe_stylegan2_model,
    #     train_dataloader=train_loader,
    #     val_dataloader=val_loader,
    #     device=device
    # )
    # trainer.train(num_epochs=100)
