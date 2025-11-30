# HPE Fine-tuning on AWS - Step 5.3 Usage Guide

This guide provides step-by-step instructions for fine-tuning Human Perceptual Embeddings (HPE) with new data on AWS infrastructure.

## Prerequisites

### 1. AWS Setup
- AWS account with appropriate permissions
- AWS CLI configured with credentials
- EC2 key pair for SSH access
- S3 bucket for data storage

### 2. Local Environment
```bash
# Install required packages
pip install boto3 torch torchvision PyYAML tqdm pandas pillow wandb

# Verify AWS credentials
aws sts get-caller-identity
```

### 3. Data Requirements
- New triplet similarity judgments (JSON format)
- Image dataset corresponding to triplet judgments
- Pre-trained HPE model (optional, for transfer learning)

## Step-by-Step Process

### Step 1: Prepare Your Data

First, prepare your new triplet data and extract features:

```bash
# Prepare triplet data and extract features
python prepare_hpe_data.py \
    --image-dir /path/to/your/images \
    --triplets /path/to/triplet_data.json \
    --output-dir ./prepared_data \
    --s3-bucket your-vlr-bucket \
    --s3-prefix hpe_data
```

**Expected triplet data format (JSON):**
```json
[
    {
        "anchor": "image_001",
        "positive": "image_045", 
        "negative": "image_123",
        "id": 1
    },
    {
        "anchor": "image_002",
        "positive": "image_067",
        "negative": "image_089", 
        "id": 2
    }
]
```

### Step 2: Configure Training Parameters

Edit the configuration file `hpe_finetune_config.yaml`:

```yaml
# Model configuration
model:
  input_dim: 2048          # ResNet feature dimension
  embedding_dim: 66        # HPE embedding dimension  
  dropout: 0.1

# Data paths (updated after Step 1)
data:
  train_triplets_path: "s3://your-bucket/hpe_data/triplets/train_triplets.json"
  val_triplets_path: "s3://your-bucket/hpe_data/triplets/val_triplets.json"
  train_features_path: "s3://your-bucket/hpe_data/features/train_features.pt"
  val_features_path: "s3://your-bucket/hpe_data/features/val_features.pt"

# Training settings
training:
  num_epochs: 50
  batch_size: 64
  learning_rate: 0.001
  triplet_margin: 1.0
  
# Pre-trained model (optional)
pretrained_model_path: "s3://your-bucket/models/hpe_base.pt"

# Output configuration  
output:
  s3_bucket: "your-bucket"
  s3_prefix: "hpe_models/finetuned"

# Weights & Biases tracking (optional)
use_wandb: true
wandb_project: "hpe-human-aligned-perception"
```

### Step 3: Deploy to AWS EC2

Launch training on EC2 using spot instances (cost-effective):

```bash
# Deploy with spot instances (recommended for cost savings)
python deploy_hpe_aws.py \
    --config hpe_finetune_config.yaml \
    --instance-type p3.2xlarge \
    --key-pair your-ec2-keypair \
    --region us-east-2 \
    --spot \
    --max-price 2.50

# For guaranteed availability (higher cost), omit --spot flag:
python deploy_hpe_aws.py \
    --config hpe_finetune_config.yaml \
    --instance-type p3.2xlarge \
    --key-pair your-ec2-keypair \
    --region us-east-2
```

### Step 4: Monitor Training Progress

```bash
# Monitor instance status
python deploy_hpe_aws.py --monitor i-1234567890abcdef0

# SSH into instance for detailed monitoring
ssh -i ~/.ssh/your-keypair.pem ec2-user@<public-ip>

# On the instance, check training progress:
screen -r hpe_training  # Attach to training session
tail -f nohup.out       # View training logs
nvidia-smi             # Check GPU usage
```

### Step 5: Retrieve Results

Training outputs are automatically saved to S3. Download results:

```bash
# Download final model
aws s3 cp s3://your-bucket/hpe_models/finetuned/hpe_best.pt ./

# Download all checkpoints
aws s3 sync s3://your-bucket/hpe_models/finetuned/ ./checkpoints/

# View training summary
cat checkpoints/training_summary.json
```

## Alternative: Local Training

For smaller datasets or debugging, run training locally:

```bash
# Run training locally (ensure GPU available)
python hpe_finetune_aws.py --config hpe_finetune_config.yaml
```

## Cost Management

### Instance Type Recommendations
- **Development/Testing**: `g4dn.xlarge` (~$0.53/hour spot, ~$1.2/hour on-demand)
- **Production Training**: `p3.2xlarge` (~$1.5/hour spot, ~$3.06/hour on-demand)
- **Large-scale Training**: `p4d.24xlarge` (~$15/hour spot, ~$32.77/hour on-demand)

### Cost Optimization Tips
1. **Use Spot Instances**: Save 50-70% compared to on-demand
2. **Auto-shutdown**: Instances automatically shutdown when training completes
3. **Monitor Billing**: Set up AWS billing alarms
4. **Right-size Instances**: Start with smaller instances for testing

### Billing Monitoring
```bash
# Set up billing alarm (run once)
aws cloudwatch put-metric-alarm \
    --alarm-name "HPE-Training-Cost-Alert" \
    --alarm-description "Alert when estimated charges exceed threshold" \
    --metric-name EstimatedCharges \
    --namespace AWS/Billing \
    --statistic Maximum \
    --period 86400 \
    --threshold 100.0 \
    --comparison-operator GreaterThanThreshold \
    --dimensions Name=Currency,Value=USD
```

## Troubleshooting

### Common Issues

**1. Instance Launch Failures**
```bash
# Check spot price history
aws ec2 describe-spot-price-history --instance-types p3.2xlarge --max-items 5

# Verify key pair exists
aws ec2 describe-key-pairs --key-names your-keypair
```

**2. Data Loading Errors**
```bash
# Verify S3 bucket access
aws s3 ls s3://your-bucket/hpe_data/

# Check triplet data format
python -c "import json; data=json.load(open('triplets.json')); print(f'Loaded {len(data)} triplets')"
```

**3. CUDA/GPU Issues**
```bash
# On EC2 instance, verify GPU availability
nvidia-smi
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

**4. Out of Memory Errors**
- Reduce batch size in config file
- Use smaller instance type for testing
- Check data loader num_workers setting

### Debugging Training
```bash
# Enable debug mode
python hpe_finetune_aws.py --config hpe_finetune_config.yaml --debug

# Monitor GPU memory usage
watch -n 1 nvidia-smi

# Check training logs
tail -f training.log
```

## Security Considerations

### IAM Permissions
Ensure your EC2 instances have appropriate IAM roles:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::your-bucket/*",
                "arn:aws:s3:::your-bucket"
            ]
        }
    ]
}
```

### Data Protection
- Enable S3 bucket encryption
- Use VPC endpoints for S3 access
- Regularly rotate access keys
- Monitor CloudTrail logs for unusual activity

## Advanced Usage

### Multi-GPU Training
For multiple GPU setup, modify the training script:

```python
# In hpe_finetune_aws.py, add DataParallel support
if torch.cuda.device_count() > 1:
    self.model = nn.DataParallel(self.model)
    logger.info(f"Using {torch.cuda.device_count()} GPUs")
```

### Hyperparameter Tuning
Use Weights & Biases sweeps for automated hyperparameter optimization:

```yaml
# wandb_sweep_config.yaml
program: hpe_finetune_aws.py
method: bayes
metric:
  name: val_loss
  goal: minimize
parameters:
  learning_rate:
    min: 0.0001
    max: 0.01
  batch_size:
    values: [32, 64, 128]
  triplet_margin:
    min: 0.5
    max: 2.0
```

### Custom Loss Functions
Extend the HPE model with additional loss terms:

```python
def compute_enhanced_triplet_loss(self, anchor, positive, negative, margin=1.0):
    # Standard triplet loss
    triplet_loss = self.compute_triplet_loss(anchor, positive, negative, margin)
    
    # Additional regularization terms
    embedding_norm_loss = torch.mean(torch.norm(anchor, p=2, dim=1))
    
    # Combine losses
    total_loss = triplet_loss + 0.01 * embedding_norm_loss
    return total_loss
```

## Results Analysis

After training, analyze the learned embeddings:

```python
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Load trained model
model = HPEModel()
checkpoint = torch.load('hpe_best.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Extract embeddings for visualization
embeddings = model(features)

# Visualize with t-SNE
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings.detach().cpu().numpy())

plt.figure(figsize=(10, 8))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6)
plt.title('HPE Learned Embeddings (t-SNE)')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.show()
```

## Next Steps

After successful fine-tuning:

1. **Integration**: Integrate fine-tuned HPE into StyleGAN2 architecture
2. **Evaluation**: Conduct human perception alignment evaluation
3. **Deployment**: Deploy for inference or further research
4. **Scaling**: Scale to larger datasets or different domains

## Support

For issues or questions:
- Check AWS CloudWatch logs for instance issues
- Monitor S3 access logs for data transfer problems  
- Use Weights & Biases dashboard for training metrics
- Review training logs for model-specific issues

Remember to terminate EC2 instances when training is complete to avoid unnecessary charges!
