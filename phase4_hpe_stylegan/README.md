# HPE-StyleGAN2: Human-Aligned Generative Perception

## Phase 3: StyleGAN2-ADA with Human Perceptual Embeddings

This repository implements Phase 3 of the Human-Aligned Generative Perception research project, integrating Human Perceptual Embeddings (HPE) with StyleGAN2-ADA architecture using the THINGS dataset.

## 🎯 Overview

The project develops generative models that align with human perceptual similarity judgments rather than just technical metrics. We integrate 66-dimensional behavioral embeddings from the THINGS dataset directly into the StyleGAN2 architecture.

### Key Features

- **HPE Integration**: 66-dimensional human perceptual embeddings from THINGS dataset
- **StyleGAN2-ADA**: Modified architecture with HPE-aware discriminator
- **Multi-objective Training**: Adversarial + HPE alignment + perceptual + triplet losses
- **AWS Deployment**: Complete cloud training infrastructure
- **THINGS Dataset**: 1,854 concepts with 4.7M human similarity judgments

## 📁 Project Structure

```
phase3_hpe_stylegan/
├── src/
│   ├── models/
│   │   ├── hpe/
│   │   │   └── hpe_core.py          # HPE embedding network
│   │   └── stylegan2/
│   │       └── hpe_stylegan2.py     # Modified StyleGAN2 architecture
│   ├── training/
│   │   └── hpe_trainer.py           # Multi-objective training loop
│   ├── data/
│   │   └── things_dataset.py        # THINGS dataset handler
│   └── utils/
│       ├── aws_utils.py             # S3 integration utilities
│       ├── evaluation.py            # Model evaluation metrics
│       └── visualization.py         # Sample generation
├── configs/
│   └── default_config.yaml          # Training configuration
├── scripts/
│   ├── train_hpe_stylegan2.py       # Main training script
│   └── aws/
│       ├── deploy.sh                # AWS deployment script
│       └── aws_deploy.py            # AWS infrastructure setup
├── requirements.txt
└── README.md
```

## 🚀 Quick Start - AWS Deployment

### Step 1: Prerequisites

```bash
# Ensure AWS CLI is configured with your credentials
aws configure

# Verify access to us-east-2 region
aws sts get-caller-identity
```

### Step 2: Deploy Infrastructure

```bash
# Clone and setup
cd phase3_hpe_stylegan

# Run automated AWS deployment
./scripts/aws/deploy.sh
```

This script will:
- Create S3 bucket for data storage
- Set up IAM roles and security groups  
- Launch EC2 instance with Deep Learning AMI
- Install dependencies and download project code
- Set up auto-shutdown after 8 hours (safety feature)

### Step 3: Start Training

```bash
# SSH into the launched instance (IP provided by deploy script)
ssh -i hpe-stylegan2-key.pem ubuntu@<PUBLIC_IP>

# Navigate to project and activate environment
cd hpe-stylegan2
conda activate hpe_stylegan

# Start training with full configuration
python scripts/train_hpe_stylegan2.py \
    --config configs/default_config.yaml \
    --s3_bucket <BUCKET_NAME> \
    --download_data \
    --upload_checkpoints \
    --epochs 200 \
    --batch_size 16

# Monitor training progress
tensorboard --logdir experiments/*/logs --host 0.0.0.0 --port 6006
```

## 🔧 Manual Setup (Local Development)

### Environment Setup

```bash
# Create conda environment
conda create -n hpe_stylegan python=3.9
conda activate hpe_stylegan

# Install PyTorch with CUDA
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu117

# Install dependencies
pip install -r requirements.txt
```

### Download THINGS Dataset

```bash
# The dataset will be automatically downloaded during training
# Or manually download to data/things/
python -c "
from src.data.things_dataset import THINGSDataset
dataset = THINGSDataset('./data/things', download=True)
print(f'Downloaded {len(dataset)} samples')
"
```

### Local Training

```bash
# Quick test run
python scripts/train_hpe_stylegan2.py \
    --epochs 10 \
    --batch_size 8 \
    --image_size 128

# Full training
python scripts/train_hpe_stylegan2.py \
    --config configs/default_config.yaml \
    --epochs 200 \
    --batch_size 16 \
    --image_size 256
```

## 🏗️ Architecture Details

### HPE Network
- **Input**: 2048-dim features from discriminator backbone
- **Architecture**: [2048 → 1024 → 512 → 256 → 66]
- **Output**: 66-dimensional human perceptual embeddings
- **Training**: Triplet loss + MSE alignment with THINGS embeddings

### Modified StyleGAN2
- **Generator**: Conditioned on HPE embeddings via style fusion
- **Discriminator**: HPE-aware with dedicated embedding branch
- **Loss Components**:
  - Adversarial loss (standard + HPE-conditioned)
  - HPE alignment loss (MSE between predicted and target HPE)
  - Perceptual loss (VGG features)
  - R1 gradient penalty
  - Triplet consistency loss

### Training Configuration

```yaml
# Key training parameters
training:
  epochs: 200
  batch_size: 16
  g_lr: 0.0002
  d_lr: 0.0002
  lambda_hpe: 1.0        # HPE alignment weight
  lambda_perceptual: 0.1 # Perceptual loss weight
  lambda_r1: 10.0        # R1 regularization
  lambda_triplet: 0.5    # Triplet consistency
```

## 📊 Monitoring and Evaluation

### TensorBoard Logging
```bash
# View training progress
tensorboard --logdir experiments/*/logs --port 6006
```

### Wandb Integration
```bash
# Login to wandb (optional)
wandb login

# Training will automatically log to wandb project "hpe-stylegan2-phase3"
```

### Key Metrics
- **HPE Similarity**: Cosine similarity between predicted and target HPE
- **FID Score**: Frechet Inception Distance for image quality
- **IS Score**: Inception Score for diversity
- **LPIPS**: Learned Perceptual Image Patch Similarity

## 🔬 Research Configuration

### Instance Recommendations

| Purpose | Instance Type | vCPUs | Memory | GPU | Cost/Hour |
|---------|--------------|-------|--------|-----|-----------|
| Development | g4dn.xlarge | 4 | 16GB | T4 16GB | $0.526 |
| Training | p3.2xlarge | 8 | 61GB | V100 16GB | $3.06 |
| Large-scale | p3.8xlarge | 32 | 244GB | 4×V100 | $12.24 |

### Dataset Specifications
- **THINGS Dataset**: 1,854 object concepts
- **Images**: 26,107 object images
- **Triplets**: 4.7M human similarity judgments
- **HPE Embeddings**: 66-dimensional SPoSE vectors
- **Splits**: 80% train, 10% validation, 10% test

### Hyperparameter Guidelines

| Parameter | Development | Full Training | Notes |
|-----------|-------------|---------------|-------|
| `image_size` | 128 | 256 | Start small for testing |
| `batch_size` | 8 | 16-32 | Adjust for GPU memory |
| `epochs` | 10 | 200+ | Full convergence needs time |
| `lambda_hpe` | 1.0 | 1.0 | HPE alignment weight |

## 🛠️ Troubleshooting

### Common Issues

**CUDA Out of Memory**
```bash
# Reduce batch size
python scripts/train_hpe_stylegan2.py --batch_size 8

# Or use gradient accumulation
# Edit config: training.gradient_accumulation_steps: 2
```

**AWS Access Issues**
```bash
# Verify credentials
aws sts get-caller-identity

# Check S3 permissions
aws s3 ls

# Verify EC2 permissions
aws ec2 describe-instances
```

**Dataset Download Fails**
```bash
# Manual download from OSF
# See src/data/things_dataset.py for URLs
# Or use smaller subset for testing
```

### AWS Cost Management

```bash
# Set up billing alerts
aws budgets create-budget --account-id <ACCOUNT_ID> \
  --budget file://budget.json

# Monitor costs
aws ce get-dimension-values --dimension SERVICE

# Auto-terminate instances
aws ec2 create-tags --resources <INSTANCE_ID> \
  --tags Key=auto-stop,Value=8hours
```

## 📈 Expected Results

### Training Metrics
- **Convergence**: ~100-200 epochs for 256×256 images
- **HPE Similarity**: Target >0.8 cosine similarity
- **FID Score**: Target <50 for human-aligned generation
- **Training Time**: ~24-48 hours on p3.2xlarge

### Research Outcomes
- Human-aligned image generation
- HPE-conditioned sample diversity
- Improved perceptual quality metrics
- Transferable HPE representations

## 🔗 Integration with Previous Phases

This Phase 3 implementation builds on:
- **Phase 1**: THINGS dataset preparation and HPE extraction
- **Phase 2**: Baseline StyleGAN2 training infrastructure
- **Future Phases**: Closed-loop human feedback and evaluation

## 📚 References

- [THINGS Dataset](https://osf.io/jum2f/): Human similarity judgments
- [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch): Base architecture
- [SPoSE](https://www.nature.com/articles/s41562-020-00951-3): Human perceptual embeddings

## 📄 License

This project is for research purposes. See individual model licenses for specific terms.

---

**⚠️ Important**: Always terminate AWS instances when not in use to avoid charges. The deployment script includes an 8-hour auto-shutdown as a safety measure.

For questions or issues, please check the troubleshooting section or review the code documentation.
