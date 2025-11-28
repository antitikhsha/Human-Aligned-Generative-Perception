"""
HPE Perceptual Loss for StyleGAN2 Integration

This module wraps the trained HPE model to provide a perceptual loss
that aligns with human perception for use in StyleGAN2 training.
"""
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add parent directory to path to import HPE model
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from hpe_model import create_hpe_model


class HPEPerceptualLoss(nn.Module):
    """
    Human-Aligned Perceptual Loss using trained HPE model.
    
    Uses the trained HPE embedding model to compute perceptual distances
    between generated and real images in a way that aligns with human judgment.
    """
    
    def __init__(self, checkpoint_path, device='cuda', embedding_dim=128):
        """
        Initialize HPE perceptual loss.
        
        Args:
            checkpoint_path: Path to trained HPE model checkpoint (.pth file)
            device: Device to run the model on
            embedding_dim: Embedding dimension (must match checkpoint)
        """
        super().__init__()
        self.device = device
        self.embedding_dim = embedding_dim
        
        print(f"Loading HPE model from: {checkpoint_path}")
        
        # Load the trained HPE model
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Get embedding dim from checkpoint if available
        if 'embedding_dim' in checkpoint:
            self.embedding_dim = checkpoint['embedding_dim']
            print(f"Using embedding_dim from checkpoint: {self.embedding_dim}")
        
        # Create model with same architecture as training
        self.hpe_model = create_hpe_model(
            embedding_dim=self.embedding_dim,
            pretrained=False,  # Don't load ImageNet weights
            device=device
        )
        
        # Load trained weights
        self.hpe_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Freeze HPE model - we don't want to update it during GAN training
        self.hpe_model.eval()
        for param in self.hpe_model.parameters():
            param.requires_grad = False
        
        print(f"HPE Perceptual Loss initialized with {self.embedding_dim}-d embeddings")
        print("HPE model frozen (no gradient updates during training)")
    
    def normalize_images(self, images):
        """
        Normalize images for HPE model input.
        
        StyleGAN2 outputs images in range [-1, 1]
        HPE model expects ImageNet normalization
        
        Args:
            images: Tensor of shape (B, 3, H, W) in range [-1, 1]
        
        Returns:
            Normalized images for HPE model
        """
        # Convert from [-1, 1] to [0, 1]
        images = (images + 1) / 2
        
        # Apply ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
        
        images = (images - mean) / std
        return images
    
    def resize_images(self, images, target_size=224):
        """
        Resize images to HPE model input size (224x224).
        
        Args:
            images: Tensor of shape (B, 3, H, W)
            target_size: Target image size (default: 224)
        
        Returns:
            Resized images
        """
        if images.shape[2] != target_size or images.shape[3] != target_size:
            images = F.interpolate(
                images,
                size=(target_size, target_size),
                mode='bilinear',
                align_corners=False
            )
        return images
    
    @torch.no_grad()
    def get_embeddings(self, images):
        """
        Get HPE embeddings for images.
        
        Args:
            images: Tensor of shape (B, 3, H, W) in range [-1, 1]
        
        Returns:
            Embeddings of shape (B, embedding_dim)
        """
        # Resize to 224x224
        images = self.resize_images(images)
        
        # Normalize for HPE model
        images = self.normalize_images(images)
        
        # Get embeddings (no gradient needed)
        embeddings = self.hpe_model(images)
        
        return embeddings
    
    def compute_loss(self, generated_images, real_images):
        """
        Compute HPE perceptual loss between generated and real images.
        
        Uses L2 distance in HPE embedding space.
        
        Args:
            generated_images: Generated images (B, 3, H, W) in range [-1, 1]
            real_images: Real images (B, 3, H, W) in range [-1, 1]
        
        Returns:
            Scalar loss value (mean distance across batch)
        """
        # Get embeddings for both generated and real images
        gen_embeddings = self.get_embeddings(generated_images)
        real_embeddings = self.get_embeddings(real_images)
        
        # Compute L2 distance in embedding space
        # This is what the model was trained to minimize
        distance = self.hpe_model.compute_distance(gen_embeddings, real_embeddings)
        
        # Return mean distance as loss
        return distance.mean()
    
    def forward(self, generated_images, real_images):
        """Forward pass - alias for compute_loss"""
        return self.compute_loss(generated_images, real_images)


def test_hpe_loss():
    """Test the HPE loss module with dummy data"""
    print("Testing HPE Perceptual Loss...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint_path = '../../results/best_hpe_model.pth'
    
    # Create HPE loss
    hpe_loss = HPEPerceptualLoss(checkpoint_path, device=device)
    
    # Create dummy images (batch_size=4, 3 channels, 256x256)
    gen_images = torch.randn(4, 3, 256, 256, device=device) * 0.5  # range ~[-1, 1]
    real_images = torch.randn(4, 3, 256, 256, device=device) * 0.5
    
    # Compute loss
    loss = hpe_loss(gen_images, real_images)
    
    print(f"Loss value: {loss.item():.4f}")
    print("✓ HPE Loss test passed!")
    
    return hpe_loss


if __name__ == '__main__':
    test_hpe_loss()
