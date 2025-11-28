"""
Quick training script for StyleGAN2 with HPE integration.

This script provides easy commands for baseline and HPE-tuned training.
"""

import os
import subprocess
import sys

def run_baseline_training():
    """
    Train StyleGAN2 without HPE (baseline).
    Expected FID: ~7.2
    """
    print("="*60)
    print("Training StyleGAN2 Baseline (No HPE)")
    print("="*60)
    
    cmd = [
        sys.executable, "stylegan2-ada-pytorch/train.py",
        "--outdir=training-runs-baseline",
        "--data=things-256",  # Use resized dataset folder
        "--gpus=1",
        "--cfg=auto",
        "--resume=ffhq256",  # Use pre-trained weights!
        "--kimg=50",  # Short fine-tuning (increase to 100-200 for final)
        "--snap=10",
        "--batch=4",  # Small batch for 6GB GPU
        "--metrics=fid50k_full"
    ]
    
    print("\nCommand:")
    print(" ".join(cmd))
    print()
    
    subprocess.run(cmd)


def run_hpe_training():
    """
    Train StyleGAN2 with HPE perceptual loss.
    Expected FID: ~7.8 (slight decrease)
    Expected human preference: 62%
    """
    print("="*60)
    print("Training StyleGAN2 with HPE Perceptual Loss")
    print("="*60)
    
    # Check if HPE checkpoint exists
    hpe_checkpoint = "results/best_hpe_model.pth"
    if not os.path.exists(hpe_checkpoint):
        print(f"ERROR: HPE checkpoint not found at {hpe_checkpoint}")
        print("Please train the HPE model first using: python train_hpe.py")
        return
    
    cmd = [
        sys.executable, "stylegan2-ada-pytorch/train.py",
        "--outdir=training-runs-hpe",
        "--data=things-256",  # Use resized dataset folder
        "--gpus=1",
        "--cfg=auto",
        "--resume=ffhq256",  # USE PRE-TRAINED WEIGHTS
        "--kimg=50",  # Short fine-tuning
        "--snap=10",
        "--batch=4",  # Small batch for 6GB GPU
        "--metrics=fid50k_full",
        f"--hpe-checkpoint={hpe_checkpoint}",
        "--hpe-weight=0.1"  # Balance between adversarial and HPE loss
    ]
    
    print("\nCommand:")
    print(" ".join(cmd))
    print()
    
    subprocess.run(cmd)


def print_usage():
    """Print usage instructions"""
    print("="*60)
    print("StyleGAN2 + HPE Training Script")
    print("="*60)
    print()
    print("Usage: python train_stylegan_hpe.py [baseline|hpe]")
    print()
    print("Options:")
    print("  baseline  - Train StyleGAN2 without HPE (baseline)")
    print("  hpe       - Train StyleGAN2 with HPE perceptual loss")
    print()
    print("Examples:")
    print("  python train_stylegan_hpe.py baseline")
    print("  python train_stylegan_hpe.py hpe")
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)
    
    mode = sys.argv[1].lower()
    
    if mode == "baseline":
        run_baseline_training()
    elif mode == "hpe":
        run_hpe_training()
    else:
        print(f"ERROR: Unknown mode '{mode}'")
        print_usage()
        sys.exit(1)
