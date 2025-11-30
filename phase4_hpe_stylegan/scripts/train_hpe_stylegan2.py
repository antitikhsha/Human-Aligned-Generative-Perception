#!/usr/bin/env python3
"""
Main Training Script for HPE-StyleGAN2 on AWS
Step-by-step execution for Phase 3 implementation

This script orchestrates the complete training pipeline:
1. Data preparation and download
2. Model initialization  
3. HPE-StyleGAN2 training
4. Model evaluation and checkpointing
5. Results upload to S3
"""

import os
import sys
import argparse
import logging
import yaml
import torch
import wandb
import boto3
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.stylegan2.hpe_stylegan2 import HPEStyleGAN2
from training.hpe_trainer import HPEStyleGAN2Trainer
from data.things_dataset import create_things_dataloaders
from utils.aws_utils import S3Manager, upload_logs_to_s3
from utils.evaluation import HPEEvaluator
from utils.visualization import generate_sample_grid

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='HPE-StyleGAN2 Training on AWS')
    
    # Training parameters
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                       help='Path to config file')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.0002,
                       help='Learning rate')
    parser.add_argument('--image_size', type=int, default=256,
                       help='Image resolution')
    
    # Model parameters
    parser.add_argument('--z_dim', type=int, default=512,
                       help='Latent dimension')
    parser.add_argument('--hpe_dim', type=int, default=66,
                       help='HPE embedding dimension')
    
    # Data parameters
    parser.add_argument('--data_root', type=str, default='./data/things',
                       help='Root directory for THINGS dataset')
    parser.add_argument('--download_data', action='store_true',
                       help='Download THINGS dataset')
    
    # AWS parameters
    parser.add_argument('--s3_bucket', type=str, default=None,
                       help='S3 bucket for data and checkpoints')
    parser.add_argument('--upload_checkpoints', action='store_true',
                       help='Upload checkpoints to S3')
    
    # Experiment parameters
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Experiment name for logging')
    parser.add_argument('--resume_from', type=str, default=None,
                       help='Resume training from checkpoint')
    parser.add_argument('--wandb_project', type=str, default='hpe-stylegan2',
                       help='Wandb project name')
    
    # Evaluation parameters
    parser.add_argument('--eval_every', type=int, default=10,
                       help='Evaluate every N epochs')
    parser.add_argument('--save_every', type=int, default=20,
                       help='Save checkpoint every N epochs')
    
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded config from {config_path}")
        return config
    except FileNotFoundError:
        logger.warning(f"Config file {config_path} not found, using defaults")
        return {}


def setup_experiment(args, config: Dict[str, Any]) -> Dict[str, Any]:
    """Set up experiment configuration and logging"""
    
    # Generate experiment name if not provided
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"hpe_stylegan2_{timestamp}"
    
    # Create experiment directory
    exp_dir = Path(f"experiments/{args.experiment_name}")
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up directories
    dirs = {
        'exp_dir': exp_dir,
        'checkpoints_dir': exp_dir / 'checkpoints',
        'logs_dir': exp_dir / 'logs',
        'outputs_dir': exp_dir / 'outputs',
        'samples_dir': exp_dir / 'samples'
    }
    
    for dir_path in dirs.values():
        if isinstance(dir_path, Path):
            dir_path.mkdir(exist_ok=True)
    
    # Combine args and config
    exp_config = {
        **config,
        **vars(args),
        **dirs
    }
    
    # Save config
    config_save_path = exp_dir / 'config.yaml'
    with open(config_save_path, 'w') as f:
        yaml.dump(exp_config, f, default_flow_style=False)
    
    logger.info(f"Experiment setup: {args.experiment_name}")
    logger.info(f"Experiment directory: {exp_dir}")
    
    return exp_config


def setup_wandb(config: Dict[str, Any]):
    """Initialize Weights & Biases logging"""
    try:
        wandb.init(
            project=config['wandb_project'],
            name=config['experiment_name'],
            config=config,
            dir=str(config['logs_dir'])
        )
        logger.info("Wandb initialized successfully")
    except Exception as e:
        logger.warning(f"Failed to initialize wandb: {e}")


def setup_data(config: Dict[str, Any]):
    """Set up data loaders"""
    logger.info("Setting up data loaders...")
    
    # Create THINGS dataset loaders
    train_loader, val_loader, test_loader = create_things_dataloaders(
        data_root=config['data_root'],
        batch_size=config['batch_size'],
        num_workers=4,
        image_size=config['image_size'],
        download=config['download_data']
    )
    
    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")
    logger.info(f"Test samples: {len(test_loader.dataset)}")
    
    return train_loader, val_loader, test_loader


def setup_model(config: Dict[str, Any]) -> HPEStyleGAN2:
    """Initialize HPE-StyleGAN2 model"""
    logger.info("Initializing HPE-StyleGAN2 model...")
    
    model = HPEStyleGAN2(
        z_dim=config['z_dim'],
        w_dim=config['z_dim'],  # Use same dimension for simplicity
        img_channels=3,
        img_resolution=config['image_size'],
        hpe_dim=config['hpe_dim']
    )
    
    # Count parameters
    gen_params = sum(p.numel() for p in model.generator.parameters())
    disc_params = sum(p.numel() for p in model.discriminator.parameters())
    total_params = gen_params + disc_params
    
    logger.info(f"Model initialized:")
    logger.info(f"  Generator parameters: {gen_params:,}")
    logger.info(f"  Discriminator parameters: {disc_params:,}")
    logger.info(f"  Total parameters: {total_params:,}")
    
    return model


def setup_trainer(model: HPEStyleGAN2, train_loader, val_loader, config: Dict[str, Any]) -> HPEStyleGAN2Trainer:
    """Initialize trainer"""
    logger.info("Setting up trainer...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Learning rates
    learning_rates = {
        'g_lr': config.get('g_lr', config['learning_rate']),
        'd_lr': config.get('d_lr', config['learning_rate'])
    }
    
    # Loss configuration
    loss_config = {
        'lambda_hpe': config.get('lambda_hpe', 1.0),
        'lambda_perceptual': config.get('lambda_perceptual', 0.1),
        'lambda_r1': config.get('lambda_r1', 10.0),
        'lambda_triplet': config.get('lambda_triplet', 0.5)
    }
    
    trainer = HPEStyleGAN2Trainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        device=device,
        learning_rates=learning_rates,
        loss_config=loss_config,
        checkpoint_dir=str(config['checkpoints_dir']),
        log_dir=str(config['logs_dir'])
    )
    
    return trainer


def setup_s3_manager(config: Dict[str, Any]) -> S3Manager:
    """Set up S3 manager for AWS integration"""
    if config.get('s3_bucket') is None:
        logger.info("No S3 bucket specified, skipping S3 setup")
        return None
    
    try:
        s3_manager = S3Manager(
            bucket_name=config['s3_bucket'],
            region=config.get('aws_region', 'us-east-2')
        )
        logger.info(f"S3 manager initialized for bucket: {config['s3_bucket']}")
        return s3_manager
    except Exception as e:
        logger.error(f"Failed to initialize S3 manager: {e}")
        return None


def train_model(trainer: HPEStyleGAN2Trainer, config: Dict[str, Any], s3_manager=None):
    """Main training loop"""
    logger.info("Starting training...")
    
    # Resume from checkpoint if specified
    if config.get('resume_from'):
        logger.info(f"Resuming from checkpoint: {config['resume_from']}")
        trainer.load_checkpoint(config['resume_from'])
    
    # Training loop
    try:
        trainer.train(
            num_epochs=config['epochs'],
            save_every=config['save_every'],
            validate_every=config['eval_every']
        )
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    
    finally:
        # Save final checkpoint
        final_checkpoint = config['checkpoints_dir'] / 'final_checkpoint.pt'
        trainer.save_checkpoint(trainer.current_epoch - 1, {})
        logger.info(f"Final checkpoint saved: {final_checkpoint}")
        
        # Upload to S3 if configured
        if s3_manager and config.get('upload_checkpoints'):
            logger.info("Uploading checkpoints to S3...")
            s3_manager.upload_directory(
                str(config['checkpoints_dir']),
                f"experiments/{config['experiment_name']}/checkpoints/"
            )


def evaluate_model(trainer: HPEStyleGAN2Trainer, test_loader, config: Dict[str, Any]):
    """Evaluate trained model"""
    logger.info("Evaluating model...")
    
    # Initialize evaluator
    evaluator = HPEEvaluator(
        model=trainer.model,
        device=trainer.device
    )
    
    # Run evaluation
    eval_results = evaluator.evaluate(test_loader)
    
    # Log results
    logger.info("Evaluation Results:")
    for metric, value in eval_results.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Save results
    results_path = config['outputs_dir'] / 'evaluation_results.yaml'
    with open(results_path, 'w') as f:
        yaml.dump(eval_results, f)
    
    # Log to wandb
    if wandb.run is not None:
        wandb.log({"eval/" + k: v for k, v in eval_results.items()})
    
    return eval_results


def generate_samples(trainer: HPEStyleGAN2Trainer, config: Dict[str, Any]):
    """Generate sample images"""
    logger.info("Generating sample images...")
    
    device = trainer.device
    model = trainer.model
    
    # Generate samples
    model.eval()
    with torch.no_grad():
        # Random samples
        z = torch.randn(16, model.generator.z_dim, device=device)
        hpe_condition = torch.randn(16, model.generator.hpe_dim, device=device)
        
        fake_images = model.generator(z, hpe_condition)
        
        # Save sample grid
        sample_path = config['samples_dir'] / 'final_samples.png'
        generate_sample_grid(fake_images, save_path=str(sample_path))
        
        logger.info(f"Sample grid saved: {sample_path}")
        
        # Log to wandb
        if wandb.run is not None:
            wandb.log({"samples": wandb.Image(str(sample_path))})


def main():
    """Main training function"""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup experiment
    exp_config = setup_experiment(args, config)
    
    # Initialize wandb
    setup_wandb(exp_config)
    
    # Setup S3 manager
    s3_manager = setup_s3_manager(exp_config)
    
    try:
        # Setup data
        train_loader, val_loader, test_loader = setup_data(exp_config)
        
        # Setup model
        model = setup_model(exp_config)
        
        # Setup trainer
        trainer = setup_trainer(model, train_loader, val_loader, exp_config)
        
        # Train model
        train_model(trainer, exp_config, s3_manager)
        
        # Evaluate model
        eval_results = evaluate_model(trainer, test_loader, exp_config)
        
        # Generate samples
        generate_samples(trainer, exp_config)
        
        logger.info("Training completed successfully!")
        
        # Upload logs to S3
        if s3_manager:
            logger.info("Uploading logs to S3...")
            upload_logs_to_s3(
                s3_manager,
                str(exp_config['logs_dir']),
                f"experiments/{exp_config['experiment_name']}/logs/"
            )
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    
    finally:
        # Clean up
        if wandb.run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()
