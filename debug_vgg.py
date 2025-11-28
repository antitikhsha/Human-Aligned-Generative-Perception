"""Quick test script to debug VGG predictions"""
import torch
from utils import load_triplets, load_image_paths, load_image_by_id, get_imagenet_transform
from vgg_perceptual_loss import create_vgg_loss

# Load first 5 triplets
triplets_df = load_triplets('odd one out/triplet_dataset/triplets_large_final_correctednc_correctedorder.csv', num_samples=5)
image_paths = load_image_paths('things image/01_image-level/image-paths.csv')

vgg_loss = create_vgg_loss(device='cpu')
transform = get_imagenet_transform()

print("\nTesting first 5 triplets:\n")
for idx, row in triplets_df.iterrows():
    img1_id = int(row['image1'])
    img2_id = int(row['image2'])
    img3_id = int(row['image3'])
    human_choice = int(row['choice'])
    
    img1 = load_image_by_id(img1_id, image_paths, 'things image/images_THINGS', transform)
    img2 = load_image_by_id(img2_id, image_paths, 'things image/images_THINGS', transform)
    img3 = load_image_by_id(img3_id, image_paths, 'things image/images_THINGS', transform)
    
    d12 = vgg_loss.compute_distance(img1, img2).item()
    d13 = vgg_loss.compute_distance(img1, img3).item()
    d23 = vgg_loss.compute_distance(img2, img3).item()
    
    avg_dist_1 = (d12 + d13) / 2
    avg_dist_2 = (d12 + d23) / 2
    avg_dist_3 = (d13 + d23) / 2
    
    if avg_dist_1 > avg_dist_2 and avg_dist_1 > avg_dist_3:
        vgg_pred = 1
    elif avg_dist_2 > avg_dist_3:
        vgg_pred = 2
    else:
        vgg_pred = 3
    
    correct = "✓" if vgg_pred == human_choice else "✗"
    
    print(f"Triplet {idx}:")
    print(f"  Images: {img1_id}, {img2_id}, {img3_id}")
    print(f"  Distances: d12={d12:.3f}, d13={d13:.3f}, d23={d23:.3f}")
    print(f"  Avg dists: img1={avg_dist_1:.3f}, img2={avg_dist_2:.3f}, img3={avg_dist_3:.3f}")
    print(f"  Human choice: {human_choice}, VGG pred: {vgg_pred} {correct}")
    print()
