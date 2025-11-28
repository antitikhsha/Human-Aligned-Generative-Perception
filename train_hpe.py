"""
Phase 2: Train Human-Aligned Perceptual Embedding (HPE)

Train ResNet18 with triplet loss on 300k THINGS triplets
Target: Kendall Tau ≥ 0.82
"""
import os
import argparse
import json
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from utils import (
    load_triplets, load_image_paths, load_image_by_id,
    get_imagenet_transform, triplet_to_anchor_pos_neg,
    prepare_data_splits, compute_kendall_tau
)
from hpe_model import create_hpe_model, TripletLoss


class TripletDataset(Dataset):
    """
    Dataset for loading THINGS triplets
    """
    
    def __init__(self, triplets_df, image_paths, image_root, transform=None):
        self.triplets = triplets_df
        self.image_paths = image_paths
        self.image_root = image_root
        self.transform = transform
    
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        row = self.triplets.iloc[idx]
        
        # Convert to anchor, positive, negative
        anchor_id, pos_id, neg_id = triplet_to_anchor_pos_neg(row)
        
        # Load images
        anchor_img = load_image_by_id(anchor_id, self.image_paths, self.image_root, self.transform)
        pos_img = load_image_by_id(pos_id, self.image_paths, self.image_root, self.transform)
        neg_img = load_image_by_id(neg_id, self.image_paths, self.image_root, self.transform)
        
        return anchor_img, pos_img, neg_img


def train_hpe(args):
    """
    Train HPE model with triplet loss
    """
    print("="*60)
    print("Phase 2: Training Human-Aligned Perceptual Embedding")
    print("="*60)
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    print(f"Using device: {device}")
    
    # Load triplets and split data
    triplets_df = load_triplets(args.triplets_csv, num_samples=args.num_triplets)
    train_df, val_df, test_df = prepare_data_splits(
        triplets_df, 
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed
    )
    
    # Load image paths
    image_paths = load_image_paths(args.image_paths_csv)
    print(f"Loaded {len(image_paths)} image paths")
    
    # Create datasets
    transform = get_imagenet_transform(resize_size=224)
    
    train_dataset = TripletDataset(train_df, image_paths, args.image_root, transform)
    val_dataset = TripletDataset(val_df, image_paths, args.image_root, transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                              shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                            shuffle=False, num_workers=args.num_workers)
    
    # Create model
    model = create_hpe_model(
        embedding_dim=args.embedding_dim,
        pretrained=args.pretrained,
        device=device
    )
    
    # Create loss and optimizer
    criterion = TripletLoss(margin=args.margin)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    print(f"\nStarting training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        # Training
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs} [Train]')
        for batch_idx, (anchor, positive, negative) in enumerate(pbar):
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)
            
            # Forward pass
            anchor_emb = model(anchor)
            pos_emb = model(positive)
            neg_emb = model(negative)
            
            # Compute loss
            loss = criterion(anchor_emb, pos_emb, neg_emb)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for anchor, positive, negative in tqdm(val_loader, desc=f'Epoch {epoch+1}/{args.epochs} [Val]'):
                anchor = anchor.to(device)
                positive = positive.to(device)
                negative = negative.to(device)
                
                anchor_emb = model(anchor)
                pos_emb = model(positive)
                neg_emb = model(negative)
                
                loss = criterion(anchor_emb, pos_emb, neg_emb)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'embedding_dim': args.embedding_dim,
                'margin': args.margin,
            }, os.path.join(args.output_dir, 'best_hpe_model.pth'))
            print(f"✓ Saved best model (val_loss: {avg_val_loss:.4f})")
        
        # Save checkpoint every N epochs
        if (epoch + 1) % args.save_freq == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, os.path.join(args.output_dir, f'hpe_model_epoch{epoch+1}.pth'))
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'args': vars(args)
    }
    
    with open(os.path.join(args.output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nTraining complete! Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {args.output_dir}/best_hpe_model.pth")
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Train HPE Model on THINGS triplets')
    
    # Data paths
    parser.add_argument('--triplets_csv', type=str,
                       default='odd one out/triplet_dataset/triplets_large_final_correctednc_correctedorder.csv')
    parser.add_argument('--image_paths_csv', type=str,
                       default='things image/01_image-level/image-paths.csv')
    parser.add_argument('--image_root', type=str,
                       default='things image/images_THINGS')
    
    # Training settings
    parser.add_argument('--num_triplets', type=int, default=300000,
                       help='Number of triplets to use for training')
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    
    # Model settings
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--pretrained', action='store_true', default=True)
    parser.add_argument('--margin', type=float, default=0.5)
    
    # Optimization
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Output
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--cpu', action='store_true')
    
    args = parser.parse_args()
    
    train_hpe(args)


if __name__ == '__main__':
    main()
