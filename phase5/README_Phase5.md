# Phase 5: Closed-Loop Human Feedback Collection

This directory contains the complete AWS infrastructure and code for Phase 5 of the VLR project, which implements a closed-loop human feedback collection system for human-aligned generative perception.

## Overview

Phase 5 consists of two main components:

### Step 5.1: Generate New Images
- Automated image generation using trained StyleGAN2+HPE models
- Distributed processing via AWS EC2 instances and SQS queues
- Scalable generation of concept variations, interpolations, and HPE-guided images

### Step 5.2: Collect New Human Triplets
- Web-based interface for human perceptual similarity judgments
- Real-time feedback collection via AWS Lambda and API Gateway
- Structured storage of human responses in DynamoDB

## Prerequisites

1. **AWS Account Setup**
   - AWS CLI configured with `VLR_project` profile
   - Sufficient permissions for EC2, S3, Lambda, DynamoDB, CloudFormation
   - Existing S3 bucket with trained StyleGAN2 models

2. **Local Dependencies**
   ```bash
   pip install boto3 pyyaml torch torchvision pillow numpy pandas matplotlib plotly
   ```

3. **Trained Models**
   - StyleGAN2+HPE checkpoint file
   - HPE embeddings (pickle file)
   - Both uploaded to your model S3 bucket

## Quick Start

### 1. Deploy Infrastructure

```bash
# Make deployment script executable
chmod +x deploy_phase5.sh

# Deploy with your model bucket name
./deploy_phase5.sh your-model-bucket-name

# Optional: Launch generation instances immediately
./deploy_phase5.sh your-model-bucket-name --launch-instances
```

### 2. Start Image Generation

```python
from phase5_aws_orchestrator import Phase5AWSOrchestrator

# Initialize orchestrator
config_path = "config/phase5_config.yaml"
orchestrator = Phase5AWSOrchestrator(config_path)

# Start generation pipeline
generation_params = {
    'model_checkpoint': 's3://your-bucket/models/stylegan2_hpe_final.pth',
    'num_concepts': 50,
    'images_per_concept': 10
}

job_id = orchestrator.start_generation_pipeline(500, generation_params)
print(f"Generation job started: {job_id}")
```

### 3. Collect Human Feedback

After deployment, share the web interface URL with participants:
```
https://[api-gateway-id].execute-api.us-east-2.amazonaws.com/prod/feedback
```

### 4. Monitor Progress

```python
# Real-time monitoring
from phase5_monitoring import Phase5Monitor

monitor = Phase5Monitor('config/phase5_config.yaml')

# Get current metrics
metrics = monitor.get_real_time_metrics()
print(json.dumps(metrics, indent=2))

# Generate dashboard
monitor.create_monitoring_dashboard()
# Open phase5_dashboard.html in browser
```

## Detailed Usage Guide

### Image Generation

#### Concept Variations
Generate multiple variations of specific concepts:

```python
task = {
    'task_type': 'concept_generation',
    'concept_name': 'airplane',
    'num_variations': 20,
    'variation_scale': 0.1,  # Controls diversity
    'job_id': 'job_001',
    'task_id': 'task_001'
}
```

#### Concept Interpolations
Create smooth transitions between concepts:

```python
task = {
    'task_type': 'interpolation_generation',
    'concept_a': 'airplane',
    'concept_b': 'bird',
    'num_steps': 15,
    'job_id': 'job_001',
    'task_id': 'task_002'
}
```

#### HPE-Guided Generation
Generate images at specific points in human perceptual space:

```python
task = {
    'task_type': 'hpe_guided_generation',
    'target_hpe': [0.1, -0.3, 0.8, ...],  # 66-dimensional HPE coordinates
    'num_samples': 10,
    'job_id': 'job_001',
    'task_id': 'task_003'
}
```

### Human Feedback Collection

#### Web Interface Features
- Triplet selection with odd-one-out task
- Progress tracking and session management
- Response time measurement
- Skip functionality for difficult triplets
- Mobile-responsive design

#### Quality Control
- Minimum/maximum response time validation
- Agreement rate analysis across participants
- Outlier detection and filtering
- User engagement tracking

#### Data Analysis
```python
# Analyze feedback patterns
analysis = monitor.analyze_feedback_patterns(days_back=7)

# Key metrics available:
# - User behavior patterns
# - Response time distributions
# - Agreement rates per triplet
# - Temporal patterns
# - Data quality metrics
```

## Architecture

### AWS Resources

- **S3 Buckets**: Generated images, human feedback data
- **DynamoDB**: Feedback responses with GSIs for analysis
- **SQS Queues**: Image generation task distribution
- **Lambda Functions**: Web interface and feedback processing
- **API Gateway**: RESTful API for web interface
- **EC2 Instances**: GPU-enabled generation workers
- **CloudWatch**: Monitoring and cost alerts

### Data Flow

1. **Generation Tasks** → SQS Queue → EC2 Workers → S3 Storage
2. **Web Interface** → API Gateway → Lambda Functions → DynamoDB
3. **Monitoring** → CloudWatch Metrics → Dashboard/Reports

### Cost Optimization

- Spot instances for generation workers (~70% cost savings)
- Auto-shutdown after idle periods
- Pay-per-request DynamoDB billing
- CloudWatch billing alarms

## Configuration

### Key Configuration Parameters

```yaml
# config/phase5_config.yaml

phase5:
  generation:
    num_generation_instances: 2
    generation_instance_type: "g4dn.xlarge"  # ~$0.53/hour
    
  feedback:
    max_triplets_per_user: 50
    min_participants_per_triplet: 5
    
cost_management:
  daily_limit_usd: 50
  auto_shutdown:
    enabled: true
    idle_timeout_minutes: 30
```

## Monitoring and Analytics

### Real-Time Metrics
- Feedback collection rates
- Image generation throughput
- System performance (CPU, Lambda duration)
- Cost tracking

### Quality Analysis
- User engagement patterns
- Response time distributions
- Agreement rates across triplets
- Data quality indicators

### Dashboard Features
- Interactive visualizations
- Cost tracking
- Performance monitoring
- Quality control metrics

## Troubleshooting

### Common Issues

1. **Lambda Timeout Errors**
   ```bash
   # Check function logs
   aws logs describe-log-groups --log-group-name-prefix "/aws/lambda/VLR-Phase5"
   ```

2. **Generation Worker Issues**
   ```bash
   # Check EC2 instance logs
   aws ec2 describe-instances --filters "Name=tag:Project,Values=VLR"
   
   # SSH to instance and check logs
   ssh -i vlr-keypair.pem ec2-user@[instance-ip]
   tail -f /var/log/cloud-init-output.log
   ```

3. **High Costs**
   ```bash
   # Check current charges
   aws cloudwatch get-metric-statistics --namespace AWS/Billing --metric-name EstimatedCharges
   
   # Terminate instances if needed
   aws ec2 terminate-instances --instance-ids i-1234567890abcdef0
   ```

### Performance Optimization

1. **Scale Generation Workers**
   ```python
   # Launch additional instances
   orchestrator.launch_generation_instances(num_instances=5)
   ```

2. **Optimize Lambda Memory**
   ```bash
   aws lambda update-function-configuration \
     --function-name VLR-Phase5-get-triplet \
     --memory-size 1024
   ```

3. **Monitor Queue Depth**
   ```python
   # Check queue metrics
   queue_attrs = orchestrator.sqs.get_queue_attributes(
       QueueUrl=queue_url,
       AttributeNames=['ApproximateNumberOfMessages']
   )
   ```

## Data Export and Analysis

### Export Feedback Data
```python
# Export to CSV for analysis
import boto3
import pandas as pd

dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('VLR-Phase5-feedback-table')

# Scan all data
response = table.scan()
df = pd.DataFrame(response['Items'])
df.to_csv('human_feedback_data.csv', index=False)
```

### Export Generated Images
```bash
# Download all generated images
aws s3 sync s3://vlr-generated-images-bucket/generated_images/ ./generated_images/
```

## Cleanup

### Quick Cleanup (Keep Data)
```bash
# Terminate EC2 instances only
aws ec2 terminate-instances --instance-ids $(aws ec2 describe-instances \
  --filters "Name=tag:Project,Values=VLR" \
  --query 'Reservations[].Instances[].InstanceId' --output text)
```

### Full Cleanup (Delete Everything)
```bash
# Delete CloudFormation stack
aws cloudformation delete-stack --stack-name VLR-Phase5-Infrastructure

# Manual cleanup if needed
python3 phase5_aws_orchestrator.py --cleanup --delete-data
```

### Cost-Safe Cleanup
```python
# Keep data but stop compute resources
orchestrator.cleanup_resources(keep_data=True)
```

## Next Steps

After collecting human feedback:

1. **Analyze Results**: Use feedback data to identify areas for model improvement
2. **Retrain Models**: Incorporate human judgments into StyleGAN2+HPE training
3. **Validate Improvements**: Generate new images and collect additional feedback
4. **Scale Collection**: Increase participant pool for more robust data

## Support

For issues or questions:
1. Check AWS CloudWatch logs for error details
2. Review configuration in `config/phase5_config.yaml`
3. Run monitoring dashboard for system health check
4. Ensure AWS credentials and permissions are correct

## File Structure

```
phase5/
├── phase5_aws_orchestrator.py          # Main orchestration logic
├── step5_1_image_generation_worker.py  # EC2 worker for image generation
├── step5_2_feedback_collection_lambda.py # Lambda functions for web interface
├── phase5_cloudformation_template.yaml # Infrastructure as code
├── phase5_monitoring.py               # Monitoring and analytics
├── deploy_phase5.sh                   # Deployment script
├── config/
│   └── phase5_config.yaml            # Configuration parameters
└── README.md                         # This file
```
