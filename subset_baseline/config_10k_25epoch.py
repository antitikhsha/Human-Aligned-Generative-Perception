"""
StyleGAN2 Training Configuration for THINGS 10k Balanced Subset
Optimized for 25 epochs on AWS g4dn.xlarge (Tesla T4, 16GB VRAM)
"""

import json
from pathlib import Path

# Training configuration
config = {
    # Dataset
    "dataset": {
        "path": "/mnt/VLR/data/THINGS_10k_balanced",
        "name": "THINGS_10k_balanced",
        "resolution": 256,
        "num_images": 10000,
        "num_concepts": 1854,
        "description": "Balanced 10k subset with even sampling across concepts"
    },
    
    # Training parameters
    "training": {
        "total_epochs": 25,
        "batch_size": 64,
        "accumulation_steps": 1,
        
        # Learning rates (adjusted for shorter training)
        "g_lr": 0.0025,  # Slightly higher for faster convergence
        "d_lr": 0.0025,
        
        # Optimization
        "optimizer": "adam",
        "beta1": 0.0,
        "beta2": 0.99,
        "epsilon": 1e-8,
        
        # Regularization
        "r1_gamma": 10.0,  # R1 regularization for discriminator
        "lazy_regularization": True,
        "lazy_reg_interval": 16,
        
        # Progressive growing (disabled for faster training)
        "progressive": False,
        
        # Mixed precision
        "fp16": True,
        "loss_scale": "dynamic"
    },
    
    # Model architecture
    "model": {
        "architecture": "stylegan2",
        "latent_dim": 512,
        "style_dim": 512,
        "num_mapping_layers": 8,
        "channel_multiplier": 2,
        "blur_kernel": [1, 3, 3, 1],
        
        # Generator
        "generator": {
            "fused_modconv": True,
            "randomize_noise": True
        },
        
        # Discriminator
        "discriminator": {
            "mbstd_group_size": 4,
            "mbstd_num_features": 1
        }
    },
    
    # Checkpointing
    "checkpoints": {
        "save_every_n_epochs": 5,
        "save_dir": "/mnt/VLR/outputs/stylegan2_10k_25epoch",
        "s3_backup": "s3://vlr-antara-outputs/stylegan2_10k_25epoch/",
        "keep_last_n": 3,  # Keep last 3 checkpoints
        "save_optimizer_state": True
    },
    
    # Logging and monitoring
    "logging": {
        "log_every_n_steps": 50,
        "log_dir": "/mnt/VLR/outputs/stylegan2_10k_25epoch/logs",
        "tensorboard": True,
        "wandb": False,  # Set to True if using Weights & Biases
        
        # What to log
        "log_metrics": [
            "loss_g",
            "loss_d",
            "loss_r1",
            "gradient_norm_g",
            "gradient_norm_d"
        ],
        
        # Sample generation during training
        "generate_samples": True,
        "sample_every_n_epochs": 5,
        "num_samples": 16,
        "sample_seed": 42
    },
    
    # Hardware
    "hardware": {
        "device": "cuda",
        "num_gpus": 1,
        "num_workers": 4,
        "pin_memory": True,
        "persistent_workers": True
    },
    
    # Reproducibility
    "seed": 42,
    
    # Estimated training time
    "estimates": {
        "steps_per_epoch": 10000 // 64,  # ~156 steps
        "total_steps": (10000 // 64) * 25,  # ~3900 steps
        "time_per_step_seconds": 0.5,  # Approximate
        "estimated_total_hours": ((10000 // 64) * 25 * 0.5) / 3600  # ~0.5 hours
    }
}

# Calculate derived values
steps_per_epoch = config["dataset"]["num_images"] // config["training"]["batch_size"]
total_steps = steps_per_epoch * config["training"]["total_epochs"]

config["training"]["steps_per_epoch"] = steps_per_epoch
config["training"]["total_steps"] = total_steps

# Save configuration
output_file = Path("/mnt/VLR/configs/stylegan2_10k_25epoch.json")
output_file.parent.mkdir(parents=True, exist_ok=True)

with open(output_file, 'w') as f:
    json.dump(config, f, indent=2)

print("Configuration saved to:", output_file)
print("\nTraining Overview:")
print(f"  Dataset: {config['dataset']['num_images']} images from {config['dataset']['num_concepts']} concepts")
print(f"  Epochs: {config['training']['total_epochs']}")
print(f"  Batch size: {config['training']['batch_size']}")
print(f"  Steps per epoch: {steps_per_epoch}")
print(f"  Total steps: {total_steps}")
print(f"  Estimated time: ~{config['estimates']['estimated_total_hours']:.1f} hours")
print(f"\nCheckpoints will be saved every {config['checkpoints']['save_every_n_epochs']} epochs")
print(f"Samples will be generated every {config['logging']['sample_every_n_epochs']} epochs")
