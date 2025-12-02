#!/usr/bin/env python3
"""
Create a balanced 10k subset from THINGS dataset for StyleGAN2 training.
Ensures even sampling across all 1,854 concepts.
"""

import os
import json
import random
from pathlib import Path
from collections import defaultdict
import shutil
from tqdm import tqdm

def create_balanced_subset(
    source_dir: str,
    output_dir: str,
    target_count: int = 10000,
    seed: int = 42
):
    """
    Create a balanced subset by sampling evenly across concepts.
    
    Args:
        source_dir: Path to THINGS/Images directory
        output_dir: Path to output directory for subset
        target_count: Target number of images (default 10000)
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Organize images by concept
    print("Step 1: Organizing images by concept...")
    concept_images = defaultdict(list)
    
    # THINGS dataset structure is typically Images/concept_name/image_files
    for concept_dir in tqdm(sorted(source_path.iterdir())):
        if concept_dir.is_dir():
            concept_name = concept_dir.name
            images = list(concept_dir.glob("*.jpg")) + list(concept_dir.glob("*.png"))
            if images:
                concept_images[concept_name] = images
    
    num_concepts = len(concept_images)
    print(f"Found {num_concepts} concepts with images")
    
    # Step 2: Calculate balanced sampling
    images_per_concept = target_count // num_concepts
    remainder = target_count % num_concepts
    
    print(f"\nStep 2: Sampling strategy:")
    print(f"  Target: {target_count} images")
    print(f"  Concepts: {num_concepts}")
    print(f"  Base images per concept: {images_per_concept}")
    print(f"  Extra images to distribute: {remainder}")
    
    # Step 3: Sample images from each concept
    print("\nStep 3: Sampling images...")
    selected_images = []
    sampling_stats = {
        'total_selected': 0,
        'concepts_processed': 0,
        'images_per_concept': {},
        'concepts_with_fewer_images': []
    }
    
    for concept, images in tqdm(concept_images.items()):
        # Determine how many images to take from this concept
        n_to_sample = images_per_concept
        
        # Give some concepts 1 extra image to reach target_count
        if sampling_stats['total_selected'] + n_to_sample < target_count:
            remaining = target_count - sampling_stats['total_selected']
            if remaining > n_to_sample and len(selected_images) < num_concepts - 1:
                n_to_sample += 1
        
        # Handle concepts with fewer images than needed
        available = len(images)
        if available < n_to_sample:
            n_to_sample = available
            sampling_stats['concepts_with_fewer_images'].append({
                'concept': concept,
                'available': available,
                'requested': images_per_concept
            })
        
        # Sample images
        sampled = random.sample(images, n_to_sample)
        selected_images.extend([(concept, img) for img in sampled])
        
        sampling_stats['images_per_concept'][concept] = n_to_sample
        sampling_stats['total_selected'] += n_to_sample
        sampling_stats['concepts_processed'] += 1
    
    print(f"\nTotal images selected: {len(selected_images)}")
    
    # Step 4: If we're short, sample more from concepts with extra images
    if len(selected_images) < target_count:
        shortage = target_count - len(selected_images)
        print(f"Need {shortage} more images to reach target...")
        
        # Find concepts with more images available
        concepts_with_extra = []
        for concept, images in concept_images.items():
            already_selected = sampling_stats['images_per_concept'][concept]
            available = len(images) - already_selected
            if available > 0:
                concepts_with_extra.append((concept, available))
        
        # Sort by most available
        concepts_with_extra.sort(key=lambda x: x[1], reverse=True)
        
        # Sample additional images
        additional_needed = shortage
        for concept, available in concepts_with_extra:
            if additional_needed <= 0:
                break
            
            n_extra = min(1, available, additional_needed)
            
            # Get images not already selected
            all_concept_images = concept_images[concept]
            already_selected_paths = [img for c, img in selected_images if c == concept]
            remaining = [img for img in all_concept_images if img not in already_selected_paths]
            
            if remaining:
                extra_sampled = random.sample(remaining, min(n_extra, len(remaining)))
                selected_images.extend([(concept, img) for img in extra_sampled])
                additional_needed -= len(extra_sampled)
                sampling_stats['images_per_concept'][concept] += len(extra_sampled)
    
    # Shuffle the final selection
    random.shuffle(selected_images)
    final_count = len(selected_images)
    
    print(f"\nFinal selection: {final_count} images")
    
    # Step 5: Copy images to output directory
    print("\nStep 4: Copying images to output directory...")
    
    for idx, (concept, img_path) in enumerate(tqdm(selected_images)):
        # Create flat structure with informative names
        output_filename = f"{concept}_{img_path.stem}{img_path.suffix}"
        output_file = output_path / output_filename
        
        shutil.copy2(img_path, output_file)
    
    # Step 6: Save metadata
    metadata = {
        'total_images': final_count,
        'target_count': target_count,
        'num_concepts': num_concepts,
        'images_per_concept_base': images_per_concept,
        'seed': seed,
        'sampling_stats': {
            'concepts_processed': sampling_stats['concepts_processed'],
            'concepts_with_fewer_images': len(sampling_stats['concepts_with_fewer_images']),
            'min_images_from_concept': min(sampling_stats['images_per_concept'].values()),
            'max_images_from_concept': max(sampling_stats['images_per_concept'].values()),
            'mean_images_per_concept': sum(sampling_stats['images_per_concept'].values()) / len(sampling_stats['images_per_concept'])
        }
    }
    
    # Save detailed stats
    stats_file = output_path / "subset_metadata.json"
    with open(stats_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save per-concept breakdown
    breakdown_file = output_path / "concept_breakdown.json"
    with open(breakdown_file, 'w') as f:
        json.dump(sampling_stats['images_per_concept'], f, indent=2)
    
    print(f"\n✓ Subset created successfully!")
    print(f"  Output directory: {output_path}")
    print(f"  Total images: {final_count}")
    print(f"  Metadata saved to: {stats_file}")
    print(f"  Concept breakdown saved to: {breakdown_file}")
    
    # Print summary statistics
    print(f"\nSampling Statistics:")
    print(f"  Concepts processed: {sampling_stats['concepts_processed']}")
    print(f"  Images per concept (min): {min(sampling_stats['images_per_concept'].values())}")
    print(f"  Images per concept (max): {max(sampling_stats['images_per_concept'].values())}")
    print(f"  Images per concept (mean): {metadata['sampling_stats']['mean_images_per_concept']:.1f}")
    
    if sampling_stats['concepts_with_fewer_images']:
        print(f"\n  Warning: {len(sampling_stats['concepts_with_fewer_images'])} concepts had fewer images than requested")
        print(f"  (This is normal for concepts with limited samples)")
    
    return output_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create balanced 10k subset from THINGS dataset")
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to THINGS Images directory (containing concept subdirectories)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output directory for subset"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10000,
        help="Target number of images (default: 10000)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    create_balanced_subset(
        source_dir=args.source,
        output_dir=args.output,
        target_count=args.count,
        seed=args.seed
    )
