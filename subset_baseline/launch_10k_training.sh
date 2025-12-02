#!/bin/bash
# Master launch script for StyleGAN2 training on 10k balanced subset
# This script will:
# 1. Create the balanced 10k subset
# 2. Set up training configuration
# 3. Launch training with proper logging
# 4. Set up automatic checkpointing to S3

set -e

echo "=========================================="
echo "StyleGAN2 Training: 10k Images, 25 Epochs"
echo "=========================================="
echo ""

# Configuration
PROJECT_ROOT="/mnt/VLR"
DATA_DIR="$PROJECT_ROOT/data"
SCRIPTS_DIR="$PROJECT_ROOT/scripts"
CONFIGS_DIR="$PROJECT_ROOT/configs"
OUTPUTS_DIR="$PROJECT_ROOT/outputs"
LOGS_DIR="$OUTPUTS_DIR/stylegan2_10k_25epoch/logs"

EXPERIMENT_NAME="stylegan2_10k_25epoch"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="${EXPERIMENT_NAME}_${TIMESTAMP}"

# AWS Configuration
AWS_PROFILE="VLR_project"
S3_DATA_BUCKET="s3://vlr-antara-data"
S3_OUTPUT_BUCKET="s3://vlr-antara-outputs"

echo "Experiment: $RUN_NAME"
echo ""

# Step 1: Create balanced subset if it doesn't exist
SUBSET_DIR="$DATA_DIR/THINGS_10k_balanced"
if [ ! -d "$SUBSET_DIR" ]; then
    echo "Step 1: Creating balanced 10k subset..."
    echo "----------------------------------------"
    
    python3 create_10k_balanced_subset.py \
        --source "$DATA_DIR/THINGS/Images" \
        --output "$SUBSET_DIR" \
        --count 10000 \
        --seed 42
    
    # Backup subset metadata to S3
    if command -v aws &> /dev/null; then
        echo "Backing up subset to S3..."
        aws s3 sync "$SUBSET_DIR" "$S3_DATA_BUCKET/THINGS_10k_balanced/" \
            --profile $AWS_PROFILE \
            --exclude "*.jpg" --exclude "*.png"  # Only sync metadata
        echo "✓ Metadata backed up"
    fi
    echo ""
else
    echo "Step 1: Using existing subset at $SUBSET_DIR"
    SUBSET_COUNT=$(find "$SUBSET_DIR" -type f \( -name "*.jpg" -o -name "*.png" \) | wc -l)
    echo "  Images: $SUBSET_COUNT"
    echo ""
fi

# Step 2: Generate training configuration
echo "Step 2: Generating training configuration..."
echo "----------------------------------------"
mkdir -p "$CONFIGS_DIR"
python3 config_10k_25epoch.py
CONFIG_FILE="$CONFIGS_DIR/stylegan2_10k_25epoch.json"
echo "✓ Configuration saved to $CONFIG_FILE"
echo ""

# Step 3: Set up output directories
echo "Step 3: Setting up output directories..."
echo "----------------------------------------"
mkdir -p "$OUTPUTS_DIR/$EXPERIMENT_NAME/checkpoints"
mkdir -p "$OUTPUTS_DIR/$EXPERIMENT_NAME/samples"
mkdir -p "$LOGS_DIR"
echo "✓ Directories created"
echo ""

# Step 4: Check GPU availability
echo "Step 4: Checking GPU availability..."
echo "----------------------------------------"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo ""
else
    echo "Warning: nvidia-smi not found. Ensure CUDA is available."
    echo ""
fi

# Step 5: Create training script
echo "Step 5: Preparing training script..."
echo "----------------------------------------"

TRAIN_SCRIPT="$SCRIPTS_DIR/train_stylegan2_10k.py"

cat > "$TRAIN_SCRIPT" << 'TRAIN_EOF'
#!/usr/bin/env python3
"""
StyleGAN2 training script for 10k balanced subset
Optimized for 25 epochs on AWS g4dn.xlarge
"""

import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from datetime import datetime
import argparse

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config JSON')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    print("=" * 60)
    print(f"StyleGAN2 Training: {config['dataset']['name']}")
    print("=" * 60)
    print(f"Dataset: {config['dataset']['num_images']} images")
    print(f"Epochs: {config['training']['total_epochs']}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Estimated time: ~{config['estimates']['estimated_total_hours']:.1f} hours")
    print("=" * 60)
    print()
    
    # TODO: Import your StyleGAN2 implementation
    # from stylegan2_baseline import StyleGAN2Generator, StyleGAN2Discriminator, train
    
    # For now, print configuration
    print("Configuration loaded successfully!")
    print(json.dumps(config, indent=2))
    
    # NOTE: Replace this with your actual training code
    # The configuration includes all necessary parameters
    
    print("\nTo complete setup, integrate your StyleGAN2 training code here.")
    print("Key config sections:")
    print("  - config['dataset']['path']: Dataset location")
    print("  - config['training']: Training hyperparameters")
    print("  - config['model']: Model architecture")
    print("  - config['checkpoints']: Checkpoint settings")
    print("  - config['logging']: Logging configuration")

if __name__ == "__main__":
    main()
TRAIN_EOF

chmod +x "$TRAIN_SCRIPT"
echo "✓ Training script created at $TRAIN_SCRIPT"
echo ""

# Step 6: Set up automatic S3 syncing
echo "Step 6: Setting up automatic S3 backup..."
echo "----------------------------------------"

SYNC_SCRIPT="$SCRIPTS_DIR/sync_outputs_to_s3.sh"
cat > "$SYNC_SCRIPT" << 'SYNC_EOF'
#!/bin/bash
# Sync outputs to S3 every 30 minutes
while true; do
    aws s3 sync /mnt/VLR/outputs/stylegan2_10k_25epoch/ \
        s3://vlr-antara-outputs/stylegan2_10k_25epoch/ \
        --profile VLR_project \
        --exclude "*.pyc" \
        --exclude "__pycache__/*"
    sleep 1800  # 30 minutes
done
SYNC_EOF

chmod +x "$SYNC_SCRIPT"
echo "✓ S3 sync script created at $SYNC_SCRIPT"
echo ""

# Step 7: Launch training
echo "Step 7: Launch training"
echo "----------------------------------------"
echo ""
echo "Ready to start training!"
echo ""
echo "To launch training in background with logging:"
echo ""
echo "  nohup python3 $TRAIN_SCRIPT \\"
echo "    --config $CONFIG_FILE \\"
echo "    > $LOGS_DIR/training_${TIMESTAMP}.log 2>&1 &"
echo ""
echo "To start automatic S3 backup (in separate terminal):"
echo ""
echo "  nohup bash $SYNC_SCRIPT > $LOGS_DIR/s3_sync.log 2>&1 &"
echo ""
echo "To monitor training:"
echo ""
echo "  tail -f $LOGS_DIR/training_${TIMESTAMP}.log"
echo ""
echo "To monitor GPU usage:"
echo ""
echo "  watch -n 1 nvidia-smi"
echo ""

# Ask if user wants to start training now
read -p "Start training now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Starting training..."
    echo ""
    
    # Start S3 sync in background
    nohup bash "$SYNC_SCRIPT" > "$LOGS_DIR/s3_sync.log" 2>&1 &
    S3_SYNC_PID=$!
    echo "✓ S3 sync started (PID: $S3_SYNC_PID)"
    
    # Start training
    nohup python3 "$TRAIN_SCRIPT" \
        --config "$CONFIG_FILE" \
        > "$LOGS_DIR/training_${TIMESTAMP}.log" 2>&1 &
    TRAIN_PID=$!
    echo "✓ Training started (PID: $TRAIN_PID)"
    
    echo ""
    echo "Processes started:"
    echo "  Training PID: $TRAIN_PID"
    echo "  S3 Sync PID: $S3_SYNC_PID"
    echo ""
    echo "Monitor with:"
    echo "  tail -f $LOGS_DIR/training_${TIMESTAMP}.log"
    echo ""
    echo "Stop training:"
    echo "  kill $TRAIN_PID"
    echo "  kill $S3_SYNC_PID"
    echo ""
else
    echo "Training not started. Use the commands above to start manually."
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
