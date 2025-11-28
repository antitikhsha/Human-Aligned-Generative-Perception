"""
Human-Aligned Perceptual Embedding (HPE) Model

ResNet18-based embedding model trained with triplet loss
"""
import torch
import torch.nn as nn
from torchvision import models


class HPEModel(nn.Module):
    """
    Human-Aligned Perceptual Embedding Model
    Based on ResNet18 with modified final layer for embeddings
    """
    
    def __init__(self, embedding_dim=128, pretrained=True):
        """
        Args:
            embedding_dim: Dimension of output embeddings
            pretrained: Whether to use pretrained ImageNet weights
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # Load pretrained ResNet18
        resnet = models.resnet18(pretrained=pretrained)
        
        # Remove final FC layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Add new embedding layer
        # ResNet18 has 512 features before the final FC
        self.embedding = nn.Sequential(
            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )
        
        print(f"HPE Model initialized with {embedding_dim}-d embeddings")
        if pretrained:
            print("Using pretrained ImageNet weights")
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor (B, 3, H, W)
            
        Returns:
            Embeddings (B, embedding_dim), L2 normalized
        """
        # Extract features
        features = self.features(x)
        features = features.view(features.size(0), -1)  # Flatten
        
        # Get embeddings
        embeddings = self.embedding(features)
        
        # L2 normalize
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def compute_distance(self, emb1, emb2):
        """
        Compute Euclidean distance between embeddings
        
        Args:
            emb1: First embedding (B, D) or (D,)
            emb2: Second embedding (B, D) or (D,)
            
        Returns:
            Distance (B,) or scalar
        """
        if emb1.dim() == 1:
            emb1 = emb1.unsqueeze(0)
        if emb2.dim() == 1:
            emb2 = emb2.unsqueeze(0)
        
        # Euclidean distance
        dist = torch.sqrt(torch.sum((emb1 - emb2) ** 2, dim=1))
        
        return dist if dist.size(0) > 1 else dist.item()
    
    def predict_triplet_choice(self, ref_emb, opt1_emb, opt2_emb):
        """
        Predict which option is more similar to reference
        
        Args:
            ref_emb: Reference embedding
            opt1_emb: Option 1 embedding
            opt2_emb: Option 2 embedding
            
        Returns:
            Predicted choice: 2 if opt1 is closer, 3 if opt2 is closer
        """
        d1 = self.compute_distance(ref_emb, opt1_emb)
        d2 = self.compute_distance(ref_emb, opt2_emb)
        
        # Smaller distance means more similar
        return 2 if d1 < d2 else 3


class TripletLoss(nn.Module):
    """
    Triplet Margin Loss
    L = max(0, d(anchor, positive) - d(anchor, negative) + margin)
    """
    
    def __init__(self, margin=0.5):
        """
        Args:
            margin: Margin for triplet loss
        """
        super().__init__()
        self.margin = margin
        print(f"Triplet Loss initialized with margin={margin}")
    
    def forward(self, anchor, positive, negative):
        """
        Compute triplet loss
        
        Args:
            anchor: Anchor embeddings (B, D)
            positive: Positive embeddings (B, D)
            negative: Negative embeddings (B, D)
            
        Returns:
            Loss value
        """
        # Compute distances
        pos_dist = torch.sum((anchor - positive) ** 2, dim=1)
        neg_dist = torch.sum((anchor - negative) ** 2, dim=1)
        
        # Triplet loss
        losses = torch.relu(pos_dist - neg_dist + self.margin)
        
        return losses.mean()


def create_hpe_model(embedding_dim=128, pretrained=True, device='cuda'):
    """
    Factory function to create HPE model
    """
    model = HPEModel(embedding_dim=embedding_dim, pretrained=pretrained)
    model = model.to(device)
    return model


if __name__ == '__main__':
    # Test
    print("Testing HPE Model...")
    
    device = 'cpu'
    model = create_hpe_model(embedding_dim=128, device=device)
    
    # Test forward pass
    dummy_input = torch.randn(4, 3, 224, 224).to(device)
    embeddings = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embeddings are L2 normalized: {torch.allclose(torch.norm(embeddings, p=2, dim=1), torch.ones(4))}")
    
    # Test triplet loss
    triplet_loss = TripletLoss(margin=0.5)
    anchor = embeddings[0:2]
    positive = embeddings[1:3]
    negative = embeddings[2:4]
    
    loss = triplet_loss(anchor, positive, negative)
    print(f"Triplet loss: {loss.item():.4f}")
    
    print("HPE Model test passed!")
