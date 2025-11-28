"""
Phase 1: VGG Baseline Evaluation

Evaluate pretrained VGG19 perceptual loss on THINGS triplets
Expected Kendall Tau: ~0.63 (showing VGG doesn't align well with humans)
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
from vgg_perceptual_loss import create_vgg_loss


def evaluate_vgg_baseline(args):
    """
    Evaluate VGG perceptual loss on THINGS triplets
    """
    print("="*60)
    print("Phase 1: VGG Baseline Evaluation")
    print("="*60)
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    print(f"Using device: {device}")
    
    # Load triplets
    triplets_df = load_triplets(args.triplets_csv, num_samples=args.num_samples)
    
    # Load image paths
    image_paths = load_image_paths(args.image_paths_csv)
    print(f"Loaded {len(image_paths)} image paths")
    
    # Create VGG perceptual loss
    vgg_loss = create_vgg_loss(device=device)
    
    # Prepare transform
    transform = get_imagenet_transform(resize_size=224)
    
    # Evaluate on triplets
    print(f"\nEvaluating VGG on {len(triplets_df)} triplets...")
    
    predictions = []
    ground_truth = []
    
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
            
            # Predict using VGG: which is more similar to img1?
            # choice "1" means img1 is odd one out
            # choice "2" means img2 is more similar to img1
            # choice "3" means img3 is more similar to img1
            
            if human_choice == 1:
                # Human said img1 is odd, so compare img2 and img3
                d21 = vgg_loss.compute_distance(img2, img1).item()
                d23 = vgg_loss.compute_distance(img2, img3).item()
                
                # If img2-img3 distance is smaller, predict choice=1 (correct)
                # If img2-img1 distance is smaller, predict choice!=1 (incorrect)
                vgg_pred = 1 if d23 < d21 else 2  # simplified prediction
            else:
                # Normal case: img1 is reference, compare img2 and img3
                d12 = vgg_loss.compute_distance(img1, img2).item()
                d13 = vgg_loss.compute_distance(img1, img3).item()
                
                # Choose the one with smaller distance
                vgg_pred = 2 if d12 < d13 else 3
            
            predictions.append(vgg_pred)
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
        'method': 'VGG19 Perceptual Loss',
        'num_triplets': len(predictions),
        'kendall_tau': float(tau),
        'accuracy': float(accuracy),
        'device': device
    }
    
    output_path = os.path.join(args.output_dir, 'vgg_baseline_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    # Also save predictions for visualization
    np.savez(os.path.join(args.output_dir, 'vgg_baseline_predictions.npz'),
             predictions=predictions,
             ground_truth=ground_truth)
    
    print("\nExpected Kendall Tau: ~0.63")
    print(f"Achieved Kendall Tau: {tau:.4f}")
    
    if tau < 0.60:
        print("⚠️ WARNING: Tau is lower than expected (< 0.60)")
    elif tau > 0.66:
        print("ℹ️ INFO: Tau is higher than expected (> 0.66)")
    else:
        print("✓ Good: Tau is in expected range (~0.63)")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate VGG Baseline on THINGS triplets')
    
    # Data paths
    parser.add_argument('--triplets_csv', type=str,
                       default='odd one out/triplet_dataset/triplets_large_final_correctednc_correctedorder.csv',
                       help='Path to triplets CSV file')
    parser.add_argument('--image_paths_csv', type=str,
                       default='things image/01_image-level/image-paths.csv',
                       help='Path to image paths CSV')
    parser.add_argument('--image_root', type=str,
                       default='things image/images_THINGS',
                       help='Root directory containing extracted THINGS images')
    
    # Evaluation settings
    parser.add_argument('--num_samples', type=int, default=10000,
                       help='Number of triplets to evaluate (None = all)')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory for results')
    parser.add_argument('--cpu', action='store_true',
                       help='Force CPU usage even if GPU is available')
    
    args = parser.parse_args()
    
    evaluate_vgg_baseline(args)


if __name__ == '__main__':
    main()
