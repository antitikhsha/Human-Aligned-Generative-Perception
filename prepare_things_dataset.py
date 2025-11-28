"""
Quick script to prepare THINGS dataset for StyleGAN2.
Resizes 800x800 images to 256x256.
"""
import os
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import shutil

def prepare_things_dataset():
    source_dir = "things image/images_THINGS/images"
    output_dir = "things-256"
    target_size = 256
    
    print(f"Preparing THINGS dataset: {source_dir} -> {output_dir}")
    print(f"Target size: {target_size}x{target_size}")
    
    # Remove old output if exists
    if os.path.exists(output_dir):
        print(f"Removing old {output_dir}...")
        shutil.rmtree(output_dir)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_files = []
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, file))
    
    print(f"Found {len(image_files)} images")
    
    # Process images
    for img_path in tqdm(image_files, desc="Resizing images"):
        # Get relative path
        rel_path = os.path.relpath(img_path, source_dir)
        
        # Create output path
        out_path = os.path.join(output_dir, rel_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        
        # Load, resize, and save
        try:
            img = Image.open(img_path).convert('RGB')
            img_resized = img.resize((target_size, target_size), Image.LANCZOS)
            
            # Save as PNG (StyleGAN2 prefers PNG)
            out_path_png = os.path.splitext(out_path)[0] + '.png'
            img_resized.save(out_path_png, 'PNG')
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    print(f"\n✓ Done! Resized images saved to: {output_dir}")
    print(f"Now you can use: --data={output_dir}")

if __name__ == "__main__":
    prepare_things_dataset()
