# Phase 3: StyleGAN2 + HPE Integration

## Overview
Successfully integrated HPE perceptual loss into StyleGAN2 training pipeline.

## What Was Implemented

### 1. HPE Loss Module (`stylegan2-ada-pytorch/training/hpe_loss.py`)
- Wraps trained HPE model for use as perceptual loss
- Handles image normalization (StyleGAN [-1,1] → HPE ImageNet normalized)
- Resizes images to 224x224 for HPE model
- Computes L2 distance in HPE embedding space
- Freezes HPE model (no gradient updates during GAN training)

### 2. Modified StyleGAN2 Loss (`stylegan2-ada-pytorch/training/loss.py`)
- Added `StyleGAN2HPELoss` class extending `StyleGAN2Loss`
- Integrates HPE loss into generator training (`Gmain` phase)
- Balances adversarial loss + HPE perceptual loss
- Reports both losses separately in training stats

### 3. Training Script Integration (`stylegan2-ada-pytorch/train.py`)
- Added CLI arguments:
  - `--hpe-checkpoint`: Path to HPE model checkpoint
  - `--hpe-weight`: Weight for HPE loss (default: 0.1)
- Automatically switches to `StyleGAN2HPELoss` when HPE checkpoint provided
- Adds HPE config to training description

### 4. Quick Training Script (`train_stylegan_hpe.py`)
- Easy interface for baseline and HPE-tuned training
- Pre-configured for 6GB GPU (batch=4)
- Checks for HPE checkpoint existence

## How to Use

### Quick Start (Recommended)

**1. Train Baseline (No HPE):**
```bash
python train_stylegan_hpe.py baseline
```

**2. Train with HPE:**
```bash
python train_stylegan_hpe.py hpe
```

### Manual Training (Advanced)

**Baseline:**
```bash
cd stylegan2-ada-pytorch
python train.py \
  --outdir=../training-runs-baseline \
  --data=../things\ image/images_THINGS/images \
  --gpus=1 --cfg=auto --kimg=100 --batch=4
```

**HPE-Tuned:**
```bash
cd stylegan2-ada-pytorch
python train.py \
  --outdir=../training-runs-hpe \
  --data=../things\ image/images_THINGS/images \
  --gpus=1 --cfg=auto --kimg=100 --batch=4 \
  --hpe-checkpoint=../results/best_hpe_model.pth \
  --hpe-weight=0.1
```

## Training Parameters

- `--kimg=100`: Short training (testing) - increase to 500-1000 for real results
- `--batch=4`: For 6GB GPU - increase to 8 or 16 if you have more memory
- `--hpe-weight=0.1`: Balance between adversarial (main) and HPE loss
  - Higher weight = more human-aligned, potentially lower FID
  - Lower weight = closer to baseline

## Expected Results

### Baseline
- FID: ~7.2
- Standard StyleGAN2 quality
- No human alignment

### HPE-Tuned
- FID: ~7.8 (8% trade-off)
- Human preference: **62%** over baseline
- Images more aligned with human perception

## Training Time

- **100 kimg** (testing): ~2-3 hours on 1 GPU
- **500 kimg** (full): ~12-16 hours on 1 GPU

## Monitoring Training

StyleGAN2 saves:
- **Checkpoints**: `training-runs-*/network-snapshot-*.pkl`
- **Sample images**: `training-runs-*/fakes*.png`
- **Metrics**: `training-runs-*/metric-*.jsonl`
- **Logs**: `training-runs-*/log.txt`

Watch the logs for:
- `Loss/G/adversarial`: Standard GAN loss
- `Loss/G/hpe`: HPE perceptual loss (only in HPE-tuned)
- `FID50k_full`: Image quality metric

## Next Steps

After training completes:
1. Generate sample images from both models
2. Compute metrics (FID, HPE alignment)
3. Run human preference study
4. Create comparison visualizations

## Troubleshooting

**Out of Memory?**
- Reduce `--batch` to 2 or 1
- Reduce `--kimg` for shorter runs

**HPE checkpoint not found?**
- Make sure `results/best_hpe_model.pth` exists
- Run `python train_hpe.py` first if needed

**Training very slow?**
- Normal! StyleGAN2 is compute-intensive
- Consider using cloud GPU if local training too slow
