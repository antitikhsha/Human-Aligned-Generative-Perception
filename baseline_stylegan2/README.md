# AWS StyleGAN2 Training with THINGS Dataset

Complete AWS infrastructure and training setup for StyleGAN2 with Human Perceptual Embeddings (HPE) using the THINGS dataset.

## Overview

This repository provides a production-ready solution for training StyleGAN2 on AWS with the following features:

- **Complete AWS Infrastructure**: CloudFormation templates for VPC, EC2, S3, IAM, and monitoring
- **THINGS Dataset Integration**: Automated download, processing, and upload of the THINGS dataset
- **Human Perceptual Embeddings**: Integration of behavioral embeddings for human-aligned generation
- **Cost Optimization**: Spot instances, auto-shutdown, and billing alarms
- **Distributed Training**: Multi-GPU support with automatic scaling
- **Monitoring & Logging**: CloudWatch metrics, TensorBoard, and real-time progress tracking

## Quick Start

### Prerequisites

1. **AWS Account** with appropriate permissions
2. **AWS CLI** configured with credentials
3. **EC2 Key Pair** in your target region
4. **Python 3.8+** for local dataset preparation (optional)

### 1. Deploy Infrastructure

```bash
# Make scripts executable
chmod +x deploy_infrastructure.sh manage_training.sh

# Deploy AWS infrastructure
./deploy_infrastructure.sh create your-key-pair-name g4dn.xlarge

# Check deployment status
./deploy_infrastructure.sh outputs
```

### 2. Prepare THINGS Dataset

```bash
# Option A: Prepare locally and upload
python prepare_things_dataset.py \
    --output-dir ./things_data \
    --s3-bucket vlr-stylegan2-data-YOUR_ACCOUNT_ID \
    --aws-region us-east-2

# Option B: Prepare directly on EC2 instance (recommended for large datasets)
# This will be done automatically when training starts
```

### 3. Launch Training

```bash
# Launch optimized training instance
./manage_training.sh launch g4dn.xlarge false true 24

# Monitor training progress
./manage_training.sh monitor i-1234567890abcdef0

# List all instances
./manage_training.sh list
```

### 4. Access Training Instance

```bash
# SSH into instance
ssh -i your-key.pem ubuntu@<instance-public-ip>

# Start training
cd stylegan2
./start_training.sh

# Monitor with Jupyter
# Browser: http://<instance-public-ip>:8888

# Monitor with TensorBoard
# Browser: http://<instance-public-ip>:6006
```

## File Structure

```
aws-stylegan2/
├── stylegan2_aws_infrastructure.yaml   # CloudFormation template
├── deploy_infrastructure.sh            # Infrastructure deployment script
├── manage_training.sh                  # Instance management script
├── stylegan2_aws_trainer.py           # Main training script
├── prepare_things_dataset.py          # Dataset preparation script
├── training_config.yaml               # Training configuration
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

## Configuration

### AWS Infrastructure (`stylegan2_aws_infrastructure.yaml`)

- **VPC & Networking**: Isolated environment with public subnet
- **EC2 Launch Template**: Optimized for ML training with Deep Learning AMI
- **S3 Buckets**: Separate buckets for data, checkpoints, and outputs
- **IAM Roles**: Secure access with minimal required permissions
- **CloudWatch Alarms**: Billing and resource monitoring

### Training Configuration (`training_config.yaml`)

Key parameters to adjust:

```yaml
# Instance-specific settings
batch_size: 8                    # Adjust based on GPU memory
mixed_precision: true           # Enables memory-efficient training
num_workers: 4                  # Data loading parallelism

# Model settings
use_hpe: true                   # Enable human perceptual embeddings
hpe_weight: 1.0                 # Weight for HPE loss component
image_size: 256                 # Training image resolution

# Training settings
learning_rate: 0.002
num_epochs: 100
save_interval: 1000             # Checkpoint frequency
```

## Instance Types and Costs

| Instance Type | GPUs | GPU Memory | RAM | Cost/Hour* | Recommended Use |
|---------------|------|------------|-----|------------|-----------------|
| g4dn.xlarge   | 1x T4 | 16GB | 16GB | $0.53 | Development/Testing |
| g4dn.2xlarge  | 1x T4 | 16GB | 32GB | $0.75 | Small-scale training |
| p3.2xlarge    | 1x V100 | 32GB | 61GB | $3.06 | Production training |
| p3.8xlarge    | 4x V100 | 128GB | 244GB | $12.24 | Large-scale/Distributed |

*Costs are approximate and vary by region. Use spot instances for 50-70% savings.

## Training Process

### 1. Dataset Preparation

The THINGS dataset contains:
- **26,107 object images** across 1,854 concepts
- **4.7M human triplet judgments** from odd-one-out tasks
- **66-dimensional behavioral embeddings** from SPoSE analysis

### 2. Model Architecture

- **StyleGAN2 Generator**: Progressive growing with style-based generation
- **Human Perceptual Embeddings**: Integrated into the loss function
- **Triplet Learning**: Respects human similarity judgments

### 3. Training Pipeline

1. **Data Loading**: Streaming from S3 with local caching
2. **HPE Integration**: Behavioral embeddings guide generation
3. **Distributed Training**: Automatic multi-GPU scaling
4. **Checkpointing**: Regular saves to S3 for resumption
5. **Monitoring**: Real-time metrics via TensorBoard and CloudWatch

## Cost Optimization

### Spot Instances

```bash
# Launch with spot instances (50-70% savings)
./manage_training.sh launch p3.2xlarge true true 12
```

### Auto-Shutdown

Instances automatically shut down after training completion or specified duration:

```yaml
# In training_config.yaml
cost_optimization:
  auto_shutdown: true
  billing_alarm_threshold: 100  # USD
```

### Storage Optimization

- **S3 Intelligent Tiering**: Automatic cost optimization
- **Lifecycle Policies**: Archive old checkpoints
- **Compression**: Efficient checkpoint storage

## Monitoring and Debugging

### CloudWatch Metrics

- CPU and GPU utilization
- Memory usage
- Billing alerts
- Custom training metrics

### TensorBoard

Access at `http://<instance-ip>:6006` for:
- Loss curves
- Generated images
- Learning rate schedules
- Model gradients

### Jupyter Notebooks

Access at `http://<instance-ip>:8888` for:
- Interactive debugging
- Data exploration
- Model analysis

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   ```yaml
   # Reduce batch size in training_config.yaml
   batch_size: 4
   mixed_precision: true
   ```

2. **Slow Data Loading**
   ```yaml
   # Increase data loader workers
   num_workers: 8
   cache_dataset: true
   ```

3. **High AWS Costs**
   ```bash
   # Use spot instances and monitoring
   ./manage_training.sh estimate p3.2xlarge 24
   ./manage_training.sh launch p3.2xlarge true true 12
   ```

4. **Network Issues**
   ```bash
   # Check security group settings
   aws ec2 describe-security-groups --group-ids sg-xxx
   ```

### Log Files

- **Training logs**: `/home/ubuntu/stylegan2/training.log`
- **System logs**: `/var/log/cloud-init-output.log`
- **AWS CloudWatch**: Logs group `/aws/ec2/stylegan2`

## Advanced Usage

### Multi-GPU Training

```bash
# Launch instance with multiple GPUs
./manage_training.sh launch p3.8xlarge false true 48

# Training will automatically use all available GPUs
python stylegan2_aws_trainer.py --config training_config.yaml --distributed
```

### Custom Dataset

1. Modify `prepare_things_dataset.py` for your dataset format
2. Update `training_config.yaml` with new parameters
3. Ensure S3 bucket contains properly formatted data

### Hyperparameter Tuning

```yaml
# In training_config.yaml
training:
  lr_scheduler: "cosine"
  gradient_penalty_weight: 10.0
  r1_penalty_weight: 10.0
  path_length_penalty_weight: 2.0
```

## Security Considerations

- **VPC Isolation**: Training runs in private subnet
- **IAM Permissions**: Minimal required access
- **Encrypted Storage**: EBS and S3 encryption enabled
- **Security Groups**: Restricted network access

## Cleanup

```bash
# Stop instances (preserves data)
./manage_training.sh stop

# Terminate instances (PERMANENT)
./manage_training.sh terminate

# Delete infrastructure (PERMANENT)
./deploy_infrastructure.sh delete
```

## Support and Contributing

### Getting Help

1. **Check logs** first: `tail -f /home/ubuntu/stylegan2/training.log`
2. **Review CloudWatch** metrics and alarms
3. **Verify configuration** in `training_config.yaml`

### Known Limitations

- **Dataset size**: Very large datasets may require additional optimization
- **Training time**: Full StyleGAN2 training can take days/weeks
- **Memory usage**: Monitor GPU memory carefully with large batch sizes

### Contributing

1. Fork the repository
2. Create feature branch
3. Test thoroughly on AWS
4. Submit pull request

## License

This project is licensed under the MIT License. The THINGS dataset has its own licensing terms.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{aws-stylegan2-things,
  title={AWS StyleGAN2 Training with THINGS Dataset},
  author={Your Name},
  year={2024},
  url={https://github.com/your-username/aws-stylegan2-things}
}

@article{hebart2019things,
  title={THINGS: A database of 1,854 object concepts and more than 26,000 naturalistic object images},
  author={Hebart, Martin N and Dickter, Adam H and Kidder, Alexis and Kwok, Wan Y and Corriveau, Anna and Van Wicklin, Caitlin and Baker, Chris I},
  journal={PLoS ONE},
  volume={14},
  number={10},
  year={2019}
}
```

## Acknowledgments

- **THINGS Database**: Martin Hebart and team
- **StyleGAN2**: NVIDIA Research
- **AWS**: Cloud infrastructure and services
