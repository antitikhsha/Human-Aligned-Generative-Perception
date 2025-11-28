"""
Phase 2: Evaluate Trained HPE Model

Evaluate trained HPE model on THINGS triplets
Target: Kendall Tau ≥ 0.82
"""
import os
import argparse
import json
import numpy as np
from tqdm import tqdm
import torch

from utils import (
    load_triplets, load_image_paths, load_image_by_id,
    get_imagenet_transform, compute_kendall_tau
)
from hpe_model import create_hpe_model


def evaluate_hpe(args):
    """
    Evaluate trained HPE model on THINGS triplets
    """
    print("="*60)
    print("Phase 2: HPE Model Evaluation")
    print("="*60)
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    print(f"Using device: {device}")
    
    # Load checkpoint
    print(f"Loading model from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    embedding_dim = checkpoint.get('embedding_dim', 128)
    margin = checkpoint.get('margin', 0.5)
    
    print(f"Model configuration:")
    print(f"  Embedding dim: {embedding_dim}")
    print(f"  Triplet margin: {margin}")
    
    # Create model and load weights
    model = create_hpe_model(embedding_dim=embedding_dim, pretrained=False, device=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load triplets
    triplets_df = load_triplets(args.triplets_csv, num_samples=args.num_samples)
    
    # Load image paths
    image_paths = load_image_paths(args.image_paths_csv)
    print(f"Loaded {len(image_paths)} image paths")
    
    # Prepare transform
    transform = get_imagenet_transform(resize_size=224)
    
    # Evaluate on triplets
    print(f"\nEvaluating HPE on {len(triplets_df)} triplets...")
    
    predictions = []
    ground_truth = []
    
    with torch.no_grad():
        for idx, row in tqdm(triplets_df.iterrows(), total=len(triplets_df)):
            try:
                # Get image IDs
                img1_id = int(row['image1'])
                img2_id = int(row['image2'])
                img3_id = int(row['image3'])
                human_choice = int(row['choice'])
                
                # Load images
                img1 = load_image_by_id(img1_id, image_paths, args.image_root, transform)
                img2 = load_image_by_id(img2_id, image_paths, args.image_root, transform)
                img3 = load_image_by_id(img3_id, image_paths, args.image_root, transform)
                
                # Get embeddings
                img1 = img1.unsqueeze(0).to(device)
                img2 = img2.unsqueeze(0).to(device)
                img3 = img3.unsqueeze(0).to(device)
                
                emb1 = model(img1).squeeze()
                emb2 = model(img2).squeeze()
                emb3 = model(img3).squeeze()
                
                # Predict using HPE
                if human_choice == 1:
                    # Human said img1 is odd, compare img2 and img3
                    d21 = model.compute_distance(emb2, emb1)
                    d23 = model.compute_distance(emb2, emb3)
                    hpe_pred = 1 if d23 < d21 else 2
                else:
                    # Normal case: img1 is reference, compare img2 and img3
                    d12 = model.compute_distance(emb1, emb2)
                    d13 = model.compute_distance(emb1, emb3)
                    hpe_pred = 2 if d12 < d13 else 3
                
                predictions.append(hpe_pred)
                ground_truth.append(human_choice)
                
            except Exception as e:
                if idx < 10:  # Only print first few errors
                    print(f"Error processing triplet {idx}: {e}")
                continue
    
    # Convert to numpy arrays
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)
    
    print(f"\nSuccessfully evaluated {len(predictions)} triplets")
    
    # Compute metrics
    print("\n" + "="*60)
    print("Results:")
    print("="*60)
    
    tau, accuracy = compute_kendall_tau(predictions, ground_truth)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    results = {
        'method': 'HPE (ResNet18 + Triplet Loss)',
        'checkpoint': args.checkpoint,
        'num_triplets': len(predictions),
        'kendall_tau': float(tau),
        'accuracy': float(accuracy),
        'embedding_dim': embedding_dim,
        'margin': margin,
        'device': device
    }
    
    output_path = os.path.join(args.output_dir, 'hpe_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    # Save predictions for visualization
    np.savez(os.path.join(args.output_dir, 'hpe_predictions.npz'),
             predictions=predictions,
             ground_truth=ground_truth)
    
    print("\nTarget Kendall Tau: ≥0.82")
    print(f"Achieved Kendall Tau: {tau:.4f}")
    
    # Compare with VGG baseline if available
    vgg_results_path = os.path.join(args.output_dir, 'vgg_baseline_results.json')
    if os.path.exists(vgg_results_path):
        with open(vgg_results_path, 'r') as f:
            vgg_results = json.load(f)
        vgg_tau = vgg_results['kendall_tau']
        improvement = ((tau - vgg_tau) / vgg_tau) * 100
        print(f"\nComparison with VGG baseline:")
        print(f"  VGG Tau: {vgg_tau:.4f}")
        print(f"  HPE Tau: {tau:.4f}")
        print(f"  Improvement: {improvement:.1f}%")
        
        if improvement >= 25:
            print("✓ EXCELLENT: Achieved ≥25% improvement!")
        elif improvement >= 20:
            print("✓ Good: Achieved ≥20% improvement")
        else:
            print("⚠️ WARNING: Improvement is less than expected")
    
    if tau >= 0.82:
        print("✓ SUCCESS: Achieved target Kendall Tau ≥0.82!")
    elif tau >= 0.75:
        print("ℹ️ Good progress, but below target (0.75-0.82)")
    else:
        print("⚠️ WARNING: Kendall Tau is below 0.75")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate HPE Model on THINGS triplets')
    
    # Data paths
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--triplets_csv', type=str,
                       default='odd one out/triplet_dataset/triplets_large_final_correctednc_correctedorder.csv')
    parser.add_argument('--image_paths_csv', type=str,
                       default='things image/01_image-level/image-paths.csv')
    parser.add_argument('--image_root', type=str,
                       default='things image/images_THINGS')
    
    # Evaluation settings
    parser.add_argument('--num_samples', type=int, default=10000,
                       help='Number of triplets to evaluate')
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--cpu', action='store_true')
    
    args = parser.parse_args()
    
    evaluate_hpe(args)


if __name__ == '__main__':
    main()
