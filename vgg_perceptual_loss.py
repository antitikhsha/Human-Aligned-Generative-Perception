"""
VGG Perceptual Loss Implementation

Uses pretrained VGG19 to extract features and compute perceptual distances
"""
import torch
import torch.nn as nn
from torchvision import models


class VGGPerceptualLoss(nn.Module):
    """
    VGG19-based perceptual loss
    Extracts features from multiple layers and computes L2 distance
    """
    
    def __init__(self, layers=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Args:
            layers: List of layer indices to extract features from
                   Default: [3, 8, 17, 26, 35] corresponding to relu1_2, relu2_2, relu3_4, relu4_4, relu5_4
            device: Device to run model on
        """
        super().__init__()
        
        if layers is None:
            # Standard perceptual loss layers
            layers = [3, 8, 17, 26, 35]
        
        self.layers = layers
        self.device = device
        
        # Load pretrained VGG19
        vgg = models.vgg19(pretrained=True).features
        
        # Extract up to the deepest layer we need
        max_layer = max(layers)
        self.vgg = nn.Sequential(*list(vgg.children())[:max_layer+1])
        
        # Freeze parameters
        for param in self.vgg.parameters():
            param.requires_grad = False
        
        self.vgg.eval()
        self.vgg.to(device)
        
        print(f"VGG Perceptual Loss initialized on {device}")
        print(f"Extracting features from layers: {layers}")
    
    def extract_features(self, x: torch.Tensor) -> dict:
        """
        Extract features from specified layers
        
        Args:
            x: Input tensor (B, 3, H, W), assumed to be ImageNet normalized
            
        Returns:
            Dictionary mapping layer index to feature tensor
        """
        features = {}
        
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.layers:
                features[i] = x
        
        return features
    
    def compute_distance(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual distance between two images
        
        Args:
            img1: First image tensor (B, 3, H, W) or (3, H, W)
            img2: Second image tensor (B, 3, H, W) or (3, H, W)
            
        Returns:
            Perceptual distance (scalar or batch of scalars)
        """
        # Ensure batch dimension
        if img1.dim() == 3:
            img1 = img1.unsqueeze(0)
        if img2.dim() == 3:
            img2 = img2.unsqueeze(0)
        
        # Move to device
        img1 = img1.to(self.device)
        img2 = img2.to(self.device)
        
        # Extract features
        with torch.no_grad():
            features1 = self.extract_features(img1)
            features2 = self.extract_features(img2)
        
        # Compute L2 distance at each layer and sum
        distance = 0.0
        for layer_idx in self.layers:
            f1 = features1[layer_idx]
            f2 = features2[layer_idx]
            
            # L2 distance (mean over all dimensions except batch)
            # Shape: (B, C, H, W) -> (B,)
            layer_dist = torch.mean((f1 - f2) ** 2, dim=[1, 2, 3])
            distance = distance + layer_dist
        
        return distance
    
    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass (alias for compute_distance)
        """
        return self.compute_distance(img1, img2)
    
    def predict_triplet_choice(self, ref: torch.Tensor, opt1: torch.Tensor, opt2: torch.Tensor) -> int:
        """
        Predict which option is more similar to reference
        
        Args:
            ref: Reference image
            opt1: Option 1 image
            opt2: Option 2 image
            
        Returns:
            Predicted choice: 2 if opt1 is closer, 3 if opt2 is closer
        """
        d1 = self.compute_distance(ref, opt1)
        d2 = self.compute_distance(ref, opt2)
        
        # Smaller distance means more similar
        # If d1 < d2, opt1 is more similar, so return 2
        # If d2 < d1, opt2 is more similar, so return 3
        return 2 if d1 < d2 else 3


def create_vgg_loss(device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Factory function to create VGG perceptual loss
    """
    return VGGPerceptualLoss(device=device)


if __name__ == '__main__':
    # Test
    print("Testing VGG Perceptual Loss...")
    
    vgg_loss = create_vgg_loss(device='cpu')
    
    # Create dummy images
    img1 = torch.randn(1, 3, 224, 224)
    img2 = torch.randn(1, 3, 224, 224)
    
    # Compute distance
    dist = vgg_loss.compute_distance(img1, img2)
    print(f"Distance: {dist.item():.4f}")
    
    # Test triplet prediction
    ref = torch.randn(3, 224, 224)
    opt1 = torch.randn(3, 224, 224)
    opt2 = torch.randn(3, 224, 224)
    
    choice = vgg_loss.predict_triplet_choice(ref, opt1, opt2)
    print(f"Predicted choice: {choice}")
    
    print("VGG Perceptual Loss test passed!")
