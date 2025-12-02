# StyleGAN2 Training: 10k Balanced Subset (25 Epochs)
## Quick Reference Guide

### Overview
This setup creates a balanced 10,000 image subset from the THINGS dataset and trains StyleGAN2 for 25 epochs on AWS g4dn.xlarge.

### Key Features
- **Balanced Sampling**: ~5-6 images per concept across all 1,854 concepts
- **Fast Training**: ~30 minutes on Tesla T4 GPU
- **Automatic Backups**: S3 sync every 30 minutes
- **Comprehensive Logging**: TensorBoard + log files

---

## Setup Instructions

### 1. Upload Scripts to AWS Instance
```bash
# On your local machine, copy scripts to AWS
scp create_10k_balanced_subset.py ec2-user@YOUR_INSTANCE:/mnt/VLR/scripts/
scp config_10k_25epoch.py ec2-user@YOUR_INSTANCE:/mnt/VLR/scripts/
scp launch_10k_training.sh ec2-user@YOUR_INSTANCE:/mnt/VLR/scripts/

# SSH into instance
ssh ec2-user@YOUR_INSTANCE
cd /mnt/VLR/scripts
```

### 2. Run Master Launch Script
```bash
# Make executable
chmod +x launch_10k_training.sh

# Run the complete setup
bash launch_10k_training.sh
```

This script will:
1. Create balanced 10k subset (if not exists)
2. Generate training configuration
3. Set up output directories
4. Check GPU availability
5. Create training script
6. Set up S3 backup
7. Optionally launch training

---

## Manual Steps (Alternative)

### Create Subset Only
```bash
python3 create_10k_balanced_subset.py \
    --source /mnt/VLR/data/THINGS/Images \
    --output /mnt/VLR/data/THINGS_10k_balanced \
    --count 10000 \
    --seed 42
```

### Generate Config Only
```bash
python3 config_10k_25epoch.py
```

### Launch Training Manually
```bash
# Start training in background
nohup python3 /mnt/VLR/scripts/train_stylegan2_10k.py \
    --config /mnt/VLR/configs/stylegan2_10k_25epoch.json \
    > /mnt/VLR/outputs/stylegan2_10k_25epoch/logs/training_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Get process ID
echo $!

# Monitor training
tail -f /mnt/VLR/outputs/stylegan2_10k_25epoch/logs/training_*.log
```

### Start S3 Backup
```bash
nohup bash /mnt/VLR/scripts/sync_outputs_to_s3.sh \
    > /mnt/VLR/outputs/stylegan2_10k_25epoch/logs/s3_sync.log 2>&1 &
```

---

## Configuration Details

### Dataset
- **Source**: THINGS dataset (26,107 images, 1,854 concepts)
- **Subset**: 10,000 images balanced across concepts
- **Resolution**: 256x256
- **Sampling**: ~5-6 images per concept

### Training Parameters
- **Epochs**: 25
- **Batch Size**: 64
- **Steps per Epoch**: ~156
- **Total Steps**: ~3,900
- **Learning Rate**: 0.0025 (both G and D)
- **Mixed Precision**: FP16 enabled
- **R1 Regularization**: gamma=10.0

### Hardware Requirements
- **Instance**: AWS g4dn.xlarge
- **GPU**: NVIDIA Tesla T4 (16GB VRAM)
- **CPU**: 4 vCPUs
- **RAM**: 16 GB
- **Storage**: ~50 GB for data + outputs

### Estimated Resources
- **Training Time**: ~0.5 hours (30 minutes)
- **Cost**: ~$0.26 @ $0.526/hour
- **Disk Usage**: 
  - Dataset subset: ~2 GB
  - Model checkpoints: ~5 GB
  - Logs + samples: ~1 GB

---

## Monitoring

### Check Training Progress
```bash
# View logs
tail -f /mnt/VLR/outputs/stylegan2_10k_25epoch/logs/training_*.log

# Watch GPU utilization
watch -n 1 nvidia-smi

# Check disk usage
df -h /mnt/VLR/outputs/

# View TensorBoard (if configured)
tensorboard --logdir /mnt/VLR/outputs/stylegan2_10k_25epoch/logs --port 6006
```

### Check Running Processes
```bash
# Find training process
ps aux | grep train_stylegan2_10k.py

# Find S3 sync process
ps aux | grep sync_outputs_to_s3.sh

# Kill processes if needed
kill <PID>
```

### Verify S3 Backups
```bash
# List checkpoints
aws s3 ls s3://vlr-antara-outputs/stylegan2_10k_25epoch/checkpoints/ \
    --profile VLR_project

# Check sync logs
tail -f /mnt/VLR/outputs/stylegan2_10k_25epoch/logs/s3_sync.log
```

---

## Output Structure

```
/mnt/VLR/
├── data/
│   └── THINGS_10k_balanced/
│       ├── concept1_img001.jpg
│       ├── concept2_img002.jpg
│       ├── ...
│       ├── subset_metadata.json
│       └── concept_breakdown.json
├── configs/
│   └── stylegan2_10k_25epoch.json
├── outputs/
│   └── stylegan2_10k_25epoch/
│       ├── checkpoints/
│       │   ├── epoch_005.pt
│       │   ├── epoch_010.pt
│       │   ├── epoch_015.pt
│       │   ├── epoch_020.pt
│       │   └── epoch_025.pt
│       ├── samples/
│       │   ├── epoch_005_samples.png
│       │   ├── ...
│       │   └── epoch_025_samples.png
│       └── logs/
│           ├── training_TIMESTAMP.log
│           ├── s3_sync.log
│           └── tensorboard/
└── scripts/
    ├── create_10k_balanced_subset.py
    ├── config_10k_25epoch.py
    ├── launch_10k_training.sh
    ├── train_stylegan2_10k.py
    └── sync_outputs_to_s3.sh
```

---

## Troubleshooting

### Issue: Out of Memory (OOM)
**Solution**: Reduce batch size in config
```bash
# Edit config file
nano /mnt/VLR/configs/stylegan2_10k_25epoch.json
# Change "batch_size": 64 to "batch_size": 32
```

### Issue: Training too slow
**Solutions**:
1. Ensure FP16 is enabled in config
2. Reduce logging frequency
3. Disable sample generation during training

### Issue: Checkpoints not saving to S3
**Check**:
```bash
# Test AWS credentials
aws s3 ls s3://vlr-antara-outputs/ --profile VLR_project

# Check sync process
ps aux | grep sync_outputs_to_s3.sh
tail -f /mnt/VLR/outputs/stylegan2_10k_25epoch/logs/s3_sync.log
```

### Issue: Dataset subset incomplete
**Rerun with verbose output**:
```bash
python3 create_10k_balanced_subset.py \
    --source /mnt/VLR/data/THINGS/Images \
    --output /mnt/VLR/data/THINGS_10k_balanced \
    --count 10000 \
    --seed 42 \
    2>&1 | tee subset_creation.log
```

---

## Next Steps After Training

### 1. Evaluate Results
```bash
# Download checkpoints
aws s3 sync s3://vlr-antara-outputs/stylegan2_10k_25epoch/checkpoints/ \
    ./local_checkpoints/ --profile VLR_project

# Generate samples for evaluation
python3 generate_samples.py --checkpoint ./local_checkpoints/epoch_025.pt
```

### 2. Compare with Full Dataset Baseline
- FID score comparison
- Visual quality assessment
- Training stability metrics

### 3. Prepare for HPE Integration
- Use epoch_025.pt as baseline
- Document hyperparameters used
- Share results with teammate working on HPE

---

## Cost Management

### Prevent Unexpected Charges
```bash
# Set up auto-shutdown after training
echo "sudo shutdown -h +60" | at now + 1 hour

# Check billing
aws ce get-cost-and-usage \
    --time-period Start=2024-12-01,End=2024-12-02 \
    --granularity DAILY \
    --metrics BlendedCost \
    --profile VLR_project
```

### Optimize Costs
- Use Spot Instances for longer runs
- Clean up old checkpoints regularly
- Delete unused datasets
- Stop instance when not training

---

## Questions?
- Check main VLR project documentation
- Review StyleGAN2 baseline code
- Contact teammate for HPE integration details
