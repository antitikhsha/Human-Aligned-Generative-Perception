#!/bin/bash

# HPE-StyleGAN2 AWS Deployment Script
# Phase 3: Step-by-Step Implementation Guide

set -e  # Exit on any error

echo "=========================================="
echo "HPE-StyleGAN2 AWS Deployment - Phase 3"
echo "Human-Aligned Generative Perception"
echo "=========================================="

# Check if AWS CLI is configured
if ! aws sts get-caller-identity > /dev/null 2>&1; then
    echo "❌ AWS CLI not configured. Please run 'aws configure' first."
    exit 1
fi

echo "✅ AWS CLI configured"

# Get current directory
PROJECT_DIR=$(pwd)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Configuration
AWS_REGION=${AWS_REGION:-us-east-2}
INSTANCE_TYPE=${INSTANCE_TYPE:-g4dn.xlarge}
KEY_NAME=${KEY_NAME:-hpe-stylegan2-key}
BUCKET_NAME_PREFIX="hpe-stylegan2"

echo "Configuration:"
echo "  AWS Region: $AWS_REGION"
echo "  Instance Type: $INSTANCE_TYPE" 
echo "  Key Name: $KEY_NAME"
echo ""

# Step 1: Setup local environment
echo "Step 1: Setting up local environment..."
cd "$PROJECT_DIR/phase3_hpe_stylegan"

# Install dependencies if not already installed
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    echo "Virtual environment exists, activating..."
    source venv/bin/activate
fi

echo "✅ Local environment ready"

# Step 2: Create S3 bucket and upload data
echo ""
echo "Step 2: Setting up S3 storage..."

# Get account ID for unique bucket naming
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
BUCKET_NAME="$BUCKET_NAME_PREFIX-${ACCOUNT_ID:0:8}"

echo "Creating S3 bucket: $BUCKET_NAME"

# Create bucket
aws s3 mb s3://$BUCKET_NAME --region $AWS_REGION 2>/dev/null || echo "Bucket already exists"

# Enable versioning
aws s3api put-bucket-versioning \
    --bucket $BUCKET_NAME \
    --versioning-configuration Status=Enabled

# Create folder structure
for folder in datasets checkpoints outputs logs configs; do
    aws s3api put-object \
        --bucket $BUCKET_NAME \
        --key "$folder/" \
        --content-length 0 > /dev/null
done

echo "✅ S3 bucket created: $BUCKET_NAME"

# Upload project code to S3
echo "Uploading project code to S3..."
aws s3 sync . s3://$BUCKET_NAME/code/ \
    --exclude "*.git*" \
    --exclude "__pycache__*" \
    --exclude "*.pyc" \
    --exclude "venv*" \
    --exclude "data*"

echo "✅ Project code uploaded"

# Step 3: Create IAM role and instance profile
echo ""
echo "Step 3: Setting up IAM permissions..."

ROLE_NAME="HPEStyleGAN2-EC2-Role"
PROFILE_NAME="$ROLE_NAME-Profile"

# Create trust policy
cat > trust-policy.json << EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Service": "ec2.amazonaws.com"
            },
            "Action": "sts:AssumeRole"
        }
    ]
}
EOF

# Create IAM role
aws iam create-role \
    --role-name $ROLE_NAME \
    --assume-role-policy-document file://trust-policy.json \
    --description "Role for HPE-StyleGAN2 EC2 instances" \
    2>/dev/null || echo "Role already exists"

# Attach policies
aws iam attach-role-policy \
    --role-name $ROLE_NAME \
    --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess \
    2>/dev/null || true

aws iam attach-role-policy \
    --role-name $ROLE_NAME \
    --policy-arn arn:aws:iam::aws:policy/CloudWatchLogsFullAccess \
    2>/dev/null || true

# Create instance profile
aws iam create-instance-profile \
    --instance-profile-name $PROFILE_NAME \
    2>/dev/null || echo "Instance profile already exists"

# Add role to instance profile
aws iam add-role-to-instance-profile \
    --instance-profile-name $PROFILE_NAME \
    --role-name $ROLE_NAME \
    2>/dev/null || true

# Wait for instance profile to be ready
echo "Waiting for IAM propagation..."
sleep 10

rm -f trust-policy.json

echo "✅ IAM role and instance profile created"

# Step 4: Create security group
echo ""
echo "Step 4: Setting up security group..."

SG_NAME="hpe-stylegan2-sg"
SG_DESC="Security group for HPE-StyleGAN2 training"

# Create security group
SG_ID=$(aws ec2 create-security-group \
    --group-name $SG_NAME \
    --description "$SG_DESC" \
    --query 'GroupId' \
    --output text 2>/dev/null) || \
SG_ID=$(aws ec2 describe-security-groups \
    --group-names $SG_NAME \
    --query 'SecurityGroups[0].GroupId' \
    --output text)

echo "Security Group ID: $SG_ID"

# Add SSH access
aws ec2 authorize-security-group-ingress \
    --group-id $SG_ID \
    --protocol tcp \
    --port 22 \
    --cidr 0.0.0.0/0 \
    2>/dev/null || echo "SSH rule already exists"

# Add Jupyter notebook access
aws ec2 authorize-security-group-ingress \
    --group-id $SG_ID \
    --protocol tcp \
    --port 8888 \
    --cidr 0.0.0.0/0 \
    2>/dev/null || echo "Jupyter rule already exists"

# Add TensorBoard access
aws ec2 authorize-security-group-ingress \
    --group-id $SG_ID \
    --protocol tcp \
    --port 6006 \
    --cidr 0.0.0.0/0 \
    2>/dev/null || echo "TensorBoard rule already exists"

echo "✅ Security group configured"

# Step 5: Create key pair
echo ""
echo "Step 5: Creating SSH key pair..."

if [ ! -f "${KEY_NAME}.pem" ]; then
    aws ec2 create-key-pair \
        --key-name $KEY_NAME \
        --query 'KeyMaterial' \
        --output text > ${KEY_NAME}.pem
    
    chmod 600 ${KEY_NAME}.pem
    echo "✅ Key pair created: ${KEY_NAME}.pem"
else
    echo "✅ Key pair already exists: ${KEY_NAME}.pem"
fi

# Step 6: Create user data script
echo ""
echo "Step 6: Preparing instance setup script..."

cat > user-data.sh << EOF
#!/bin/bash

# HPE-StyleGAN2 Instance Setup Script
exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1

echo "Starting HPE-StyleGAN2 setup..."

# Update system
apt-get update -y
apt-get upgrade -y

# Install additional dependencies
apt-get install -y git vim htop tmux tree zip unzip

# Set up for ubuntu user
su - ubuntu -c "
# Set up environment
export AWS_DEFAULT_REGION=$AWS_REGION
export BUCKET_NAME=$BUCKET_NAME

# Create project directory
mkdir -p /home/ubuntu/hpe-stylegan2
cd /home/ubuntu/hpe-stylegan2

# Download project from S3
aws s3 sync s3://$BUCKET_NAME/code/ .

# Create conda environment
conda create -n hpe_stylegan python=3.9 -y
source activate hpe_stylegan

# Install PyTorch with CUDA
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu117

# Install project dependencies
pip install -r requirements.txt

# Configure environment
echo 'export PYTHONPATH=/home/ubuntu/hpe-stylegan2/src:\$PYTHONPATH' >> ~/.bashrc
echo 'conda activate hpe_stylegan' >> ~/.bashrc
echo 'export AWS_DEFAULT_REGION=$AWS_REGION' >> ~/.bashrc
echo 'export BUCKET_NAME=$BUCKET_NAME' >> ~/.bashrc

# Create directories
mkdir -p {data,experiments,checkpoints,logs,outputs}

# Download THINGS dataset (this will be done in the training script)
echo 'Ready for training!'
"

# Set up auto-shutdown (8 hours)
echo '#!/bin/bash' > /home/ubuntu/auto_shutdown.sh
echo 'shutdown -h +480' >> /home/ubuntu/auto_shutdown.sh
chmod +x /home/ubuntu/auto_shutdown.sh
crontab -l -u ubuntu 2>/dev/null; echo "@reboot /home/ubuntu/auto_shutdown.sh" | crontab -u ubuntu -

echo "HPE-StyleGAN2 setup completed!"
EOF

# Base64 encode user data
USER_DATA=$(base64 -w 0 user-data.sh)

echo "✅ User data script prepared"

# Step 7: Launch EC2 instance
echo ""
echo "Step 7: Launching EC2 instance..."

# Get Deep Learning AMI ID for the region
AMI_ID="ami-0c02fb55956c7d316"  # Deep Learning AMI (Ubuntu 20.04) us-east-2

echo "Launching $INSTANCE_TYPE instance with Deep Learning AMI..."
echo "⚠️  Estimated cost: \$$(python3 -c "
costs = {
    'g4dn.xlarge': 0.526,
    'p3.2xlarge': 3.06,
    'p3.8xlarge': 12.24
}
print(f'{costs.get(\"$INSTANCE_TYPE\", 0.5):.3f}')")/hour"

read -p "Continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Deployment cancelled"
    exit 1
fi

INSTANCE_ID=$(aws ec2 run-instances \
    --image-id $AMI_ID \
    --count 1 \
    --instance-type $INSTANCE_TYPE \
    --key-name $KEY_NAME \
    --security-group-ids $SG_ID \
    --iam-instance-profile Name=$PROFILE_NAME \
    --user-data $USER_DATA \
    --block-device-mappings '[{
        "DeviceName": "/dev/sda1",
        "Ebs": {
            "VolumeSize": 100,
            "VolumeType": "gp3",
            "DeleteOnTermination": true
        }
    }]' \
    --tag-specifications '[{
        "ResourceType": "instance",
        "Tags": [
            {"Key": "Name", "Value": "hpe-stylegan2-training"},
            {"Key": "Project", "Value": "HPE-StyleGAN2"},
            {"Key": "Phase", "Value": "3"}
        ]
    }]' \
    --query 'Instances[0].InstanceId' \
    --output text)

echo "✅ Instance launched: $INSTANCE_ID"

# Wait for instance to be running
echo "Waiting for instance to be running..."
aws ec2 wait instance-running --instance-ids $INSTANCE_ID

# Get public IP
PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids $INSTANCE_ID \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo ""
echo "=========================================="
echo "🚀 DEPLOYMENT SUCCESSFUL!"
echo "=========================================="
echo ""
echo "Instance Details:"
echo "  Instance ID: $INSTANCE_ID"
echo "  Instance Type: $INSTANCE_TYPE"
echo "  Public IP: $PUBLIC_IP"
echo "  SSH Key: ${KEY_NAME}.pem"
echo "  S3 Bucket: $BUCKET_NAME"
echo ""
echo "Access Commands:"
echo "  SSH: ssh -i ${KEY_NAME}.pem ubuntu@$PUBLIC_IP"
echo "  Monitor setup: ssh -i ${KEY_NAME}.pem ubuntu@$PUBLIC_IP 'tail -f /var/log/user-data.log'"
echo ""
echo "Training Commands (run after SSH):"
echo "  cd hpe-stylegan2"
echo "  conda activate hpe_stylegan"
echo "  python scripts/train_hpe_stylegan2.py --config configs/default_config.yaml --s3_bucket $BUCKET_NAME"
echo ""
echo "Jupyter Notebook (after setup):"
echo "  URL: http://$PUBLIC_IP:8888"
echo "  TensorBoard: http://$PUBLIC_IP:6006"
echo ""
echo "⚠️  Remember to terminate the instance when done:"
echo "  aws ec2 terminate-instances --instance-ids $INSTANCE_ID"
echo ""

# Clean up temporary files
rm -f user-data.sh

# Save deployment info
cat > deployment_info.txt << EOF
HPE-StyleGAN2 AWS Deployment Information
Generated: $(date)

Instance ID: $INSTANCE_ID
Instance Type: $INSTANCE_TYPE
Public IP: $PUBLIC_IP
SSH Key: ${KEY_NAME}.pem
S3 Bucket: $BUCKET_NAME
Security Group: $SG_ID
IAM Role: $ROLE_NAME

SSH Command:
ssh -i ${KEY_NAME}.pem ubuntu@$PUBLIC_IP

Terminate Command:
aws ec2 terminate-instances --instance-ids $INSTANCE_ID
EOF

echo "📝 Deployment info saved to: deployment_info.txt"
echo ""
echo "🎯 Next Steps:"
echo "1. Wait 5-10 minutes for instance setup to complete"
echo "2. SSH into the instance and verify setup"
echo "3. Start training with the provided command"
echo "4. Monitor progress via SSH or TensorBoard"
echo "5. Don't forget to terminate when done!"
