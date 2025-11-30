"""
AWS Configuration and Deployment for HPE-StyleGAN2
Step-by-step deployment on AWS EC2 with S3 integration
"""

import boto3
import json
import time
import subprocess
import os
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class AWSConfig:
    """AWS Configuration for HPE-StyleGAN2 deployment"""
    
    # EC2 Instance configurations
    INSTANCE_CONFIGS = {
        'development': {
            'instance_type': 'g4dn.xlarge',
            'vcpus': 4,
            'memory_gb': 16,
            'gpu': 'T4 16GB',
            'cost_per_hour': 0.526,
            'storage_gb': 125,
            'use_case': 'Development and testing'
        },
        'training': {
            'instance_type': 'p3.2xlarge', 
            'vcpus': 8,
            'memory_gb': 61,
            'gpu': 'V100 16GB',
            'cost_per_hour': 3.06,
            'storage_gb': 125,
            'use_case': 'Full model training'
        },
        'large_training': {
            'instance_type': 'p3.8xlarge',
            'vcpus': 32,
            'memory_gb': 244,
            'gpu': '4x V100 16GB',
            'cost_per_hour': 12.24,
            'storage_gb': 125,
            'use_case': 'Large-scale multi-GPU training'
        }
    }
    
    # Deep Learning AMI
    DEEP_LEARNING_AMI = {
        'ami_id': 'ami-0c02fb55956c7d316',  # Deep Learning AMI (Ubuntu 20.04)
        'name': 'Deep Learning AMI GPU PyTorch 1.13.1 (Ubuntu 20.04)',
        'python_version': '3.9',
        'pytorch_version': '1.13.1',
        'cuda_version': '11.7'
    }
    
    # Security group configuration
    SECURITY_GROUP_CONFIG = {
        'group_name': 'hpe-stylegan2-sg',
        'description': 'Security group for HPE-StyleGAN2 training',
        'ingress_rules': [
            {
                'protocol': 'tcp',
                'port': 22,
                'cidr': '0.0.0.0/0',
                'description': 'SSH access'
            },
            {
                'protocol': 'tcp', 
                'port': 8888,
                'cidr': '0.0.0.0/0',
                'description': 'Jupyter notebook'
            },
            {
                'protocol': 'tcp',
                'port': 6006,
                'cidr': '0.0.0.0/0', 
                'description': 'TensorBoard'
            }
        ]
    }
    
    # S3 bucket configuration
    S3_CONFIG = {
        'bucket_name': 'hpe-stylegan2-{user_id}',  # Will be formatted with user ID
        'regions': {
            'primary': 'us-east-2',    # Ohio - your configured region
            'backup': 'us-west-2'      # Oregon - for redundancy
        },
        'folders': {
            'datasets': 'datasets/',
            'checkpoints': 'checkpoints/',
            'outputs': 'outputs/',
            'logs': 'logs/',
            'configs': 'configs/'
        }
    }


class AWSDeployer:
    """Handles AWS deployment for HPE-StyleGAN2 project"""
    
    def __init__(self, region: str = 'us-east-2', profile: str = 'default'):
        """
        Initialize AWS deployer
        
        Args:
            region: AWS region
            profile: AWS profile name
        """
        self.region = region
        self.profile = profile
        
        # Initialize AWS clients
        self.session = boto3.Session(profile_name=profile, region_name=region)
        self.ec2 = self.session.client('ec2')
        self.s3 = self.session.client('s3')
        self.iam = self.session.client('iam')
        
        # Get account ID for unique naming
        self.account_id = self.session.client('sts').get_caller_identity()['Account']
        
        logger.info(f"Initialized AWS deployer for region: {region}")
    
    def create_s3_bucket(self) -> str:
        """Create S3 bucket for dataset and model storage"""
        bucket_name = AWSConfig.S3_CONFIG['bucket_name'].format(user_id=self.account_id[:8])
        
        try:
            # Check if bucket exists
            self.s3.head_bucket(Bucket=bucket_name)
            logger.info(f"S3 bucket {bucket_name} already exists")
            return bucket_name
            
        except self.s3.exceptions.NoSuchBucket:
            # Create bucket
            logger.info(f"Creating S3 bucket: {bucket_name}")
            
            if self.region == 'us-east-1':
                self.s3.create_bucket(Bucket=bucket_name)
            else:
                self.s3.create_bucket(
                    Bucket=bucket_name,
                    CreateBucketConfiguration={'LocationConstraint': self.region}
                )
            
            # Enable versioning
            self.s3.put_bucket_versioning(
                Bucket=bucket_name,
                VersioningConfiguration={'Status': 'Enabled'}
            )
            
            # Create folder structure
            for folder in AWSConfig.S3_CONFIG['folders'].values():
                self.s3.put_object(Bucket=bucket_name, Key=folder)
            
            logger.info(f"S3 bucket {bucket_name} created successfully")
            return bucket_name
    
    def create_iam_role(self) -> str:
        """Create IAM role for EC2 instances"""
        role_name = 'HPEStyleGAN2-EC2-Role'
        
        # Trust policy for EC2
        trust_policy = {
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
        
        try:
            # Check if role exists
            self.iam.get_role(RoleName=role_name)
            logger.info(f"IAM role {role_name} already exists")
            
        except self.iam.exceptions.NoSuchEntityException:
            # Create role
            logger.info(f"Creating IAM role: {role_name}")
            
            self.iam.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(trust_policy),
                Description='Role for HPE-StyleGAN2 EC2 instances'
            )
            
            # Attach policies
            policies = [
                'arn:aws:iam::aws:policy/AmazonS3FullAccess',
                'arn:aws:iam::aws:policy/CloudWatchLogsFullAccess'
            ]
            
            for policy in policies:
                self.iam.attach_role_policy(
                    RoleName=role_name,
                    PolicyArn=policy
                )
        
        # Create instance profile
        profile_name = f'{role_name}-Profile'
        try:
            self.iam.get_instance_profile(InstanceProfileName=profile_name)
        except self.iam.exceptions.NoSuchEntityException:
            self.iam.create_instance_profile(InstanceProfileName=profile_name)
            self.iam.add_role_to_instance_profile(
                InstanceProfileName=profile_name,
                RoleName=role_name
            )
            # Wait for instance profile to be ready
            time.sleep(10)
        
        return profile_name
    
    def create_security_group(self) -> str:
        """Create security group for EC2 instances"""
        sg_config = AWSConfig.SECURITY_GROUP_CONFIG
        
        try:
            # Check if security group exists
            response = self.ec2.describe_security_groups(
                Filters=[
                    {'Name': 'group-name', 'Values': [sg_config['group_name']]}
                ]
            )
            
            if response['SecurityGroups']:
                sg_id = response['SecurityGroups'][0]['GroupId']
                logger.info(f"Security group {sg_config['group_name']} already exists: {sg_id}")
                return sg_id
        
        except Exception:
            pass
        
        # Create security group
        logger.info(f"Creating security group: {sg_config['group_name']}")
        
        response = self.ec2.create_security_group(
            GroupName=sg_config['group_name'],
            Description=sg_config['description']
        )
        sg_id = response['GroupId']
        
        # Add ingress rules
        ingress_rules = []
        for rule in sg_config['ingress_rules']:
            ingress_rules.append({
                'IpProtocol': rule['protocol'],
                'FromPort': rule['port'],
                'ToPort': rule['port'],
                'IpRanges': [{'CidrIp': rule['cidr'], 'Description': rule['description']}]
            })
        
        self.ec2.authorize_security_group_ingress(
            GroupId=sg_id,
            IpPermissions=ingress_rules
        )
        
        logger.info(f"Security group {sg_id} created successfully")
        return sg_id
    
    def create_key_pair(self, key_name: str = 'hpe-stylegan2-key') -> str:
        """Create EC2 key pair"""
        try:
            # Check if key pair exists
            self.ec2.describe_key_pairs(KeyNames=[key_name])
            logger.info(f"Key pair {key_name} already exists")
            return key_name
            
        except self.ec2.exceptions.ClientError:
            # Create key pair
            logger.info(f"Creating key pair: {key_name}")
            
            response = self.ec2.create_key_pair(KeyName=key_name)
            
            # Save private key
            key_file = Path(f'{key_name}.pem')
            with open(key_file, 'w') as f:
                f.write(response['KeyMaterial'])
            
            # Set proper permissions
            os.chmod(key_file, 0o600)
            
            logger.info(f"Key pair {key_name} created and saved to {key_file}")
            return key_name
    
    def launch_instance(
        self, 
        instance_config: str = 'development',
        instance_name: str = 'hpe-stylegan2-instance'
    ) -> str:
        """Launch EC2 instance for training"""
        
        config = AWSConfig.INSTANCE_CONFIGS[instance_config]
        ami_config = AWSConfig.DEEP_LEARNING_AMI
        
        # Prepare dependencies
        bucket_name = self.create_s3_bucket()
        instance_profile = self.create_iam_role()
        security_group_id = self.create_security_group()
        key_name = self.create_key_pair()
        
        # User data script for instance setup
        user_data = self.generate_user_data_script(bucket_name)
        
        logger.info(f"Launching {config['instance_type']} instance...")
        logger.info(f"Estimated cost: ${config['cost_per_hour']:.3f}/hour")
        
        # Launch instance
        response = self.ec2.run_instances(
            ImageId=ami_config['ami_id'],
            MinCount=1,
            MaxCount=1,
            InstanceType=config['instance_type'],
            KeyName=key_name,
            SecurityGroupIds=[security_group_id],
            IamInstanceProfile={'Name': instance_profile},
            UserData=user_data,
            BlockDeviceMappings=[
                {
                    'DeviceName': '/dev/sda1',
                    'Ebs': {
                        'VolumeSize': config['storage_gb'],
                        'VolumeType': 'gp3',
                        'DeleteOnTermination': True
                    }
                }
            ],
            TagSpecifications=[
                {
                    'ResourceType': 'instance',
                    'Tags': [
                        {'Key': 'Name', 'Value': instance_name},
                        {'Key': 'Project', 'Value': 'HPE-StyleGAN2'},
                        {'Key': 'Purpose', 'Value': config['use_case']}
                    ]
                }
            ]
        )
        
        instance_id = response['Instances'][0]['InstanceId']
        
        logger.info(f"Instance {instance_id} launched successfully")
        logger.info(f"Waiting for instance to be ready...")
        
        # Wait for instance to be running
        waiter = self.ec2.get_waiter('instance_running')
        waiter.wait(InstanceIds=[instance_id])
        
        # Get public IP
        response = self.ec2.describe_instances(InstanceIds=[instance_id])
        public_ip = response['Reservations'][0]['Instances'][0]['PublicIpAddress']
        
        logger.info(f"Instance ready! Public IP: {public_ip}")
        logger.info(f"SSH command: ssh -i {key_name}.pem ubuntu@{public_ip}")
        
        return instance_id
    
    def generate_user_data_script(self, bucket_name: str) -> str:
        """Generate user data script for EC2 instance setup"""
        script = f"""#!/bin/bash
# HPE-StyleGAN2 Instance Setup Script

# Set up logging
exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1
echo "Starting HPE-StyleGAN2 setup..."

# Update system
apt-get update -y
apt-get upgrade -y

# Install additional dependencies
apt-get install -y git vim htop tmux tree zip unzip

# Set up conda environment
su - ubuntu -c "
# Clone project repository (you'll need to update this with your repo URL)
cd /home/ubuntu
git clone https://github.com/YOUR_USERNAME/hpe-stylegan2.git || echo 'Repository not found, creating directory structure'
mkdir -p hpe-stylegan2

# Create conda environment
conda create -n hpe_stylegan python=3.9 -y
source activate hpe_stylegan

# Install PyTorch and dependencies
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117

# Install additional packages
pip install boto3 wandb tensorboard pillow numpy scipy matplotlib pandas scikit-learn tqdm click rich opencv-python

# Configure AWS CLI (credentials should be available via IAM role)
aws configure set default.region {self.region}

# Download project data from S3
mkdir -p /home/ubuntu/data
aws s3 sync s3://{bucket_name}/datasets/ /home/ubuntu/data/

# Set up project environment
echo 'export PYTHONPATH=/home/ubuntu/hpe-stylegan2/src:$PYTHONPATH' >> ~/.bashrc
echo 'conda activate hpe_stylegan' >> ~/.bashrc

# Create directories
mkdir -p /home/ubuntu/hpe-stylegan2/{{checkpoints,logs,outputs}}
"

# Set up automatic shutdown (safety measure)
echo '#!/bin/bash' > /home/ubuntu/auto_shutdown.sh
echo 'shutdown -h +480' >> /home/ubuntu/auto_shutdown.sh  # Shutdown after 8 hours
chmod +x /home/ubuntu/auto_shutdown.sh

# Add to crontab to run at startup
crontab -l -u ubuntu 2>/dev/null; echo "@reboot /home/ubuntu/auto_shutdown.sh" | crontab -u ubuntu -

echo "HPE-StyleGAN2 setup completed!"
"""
        return script
    
    def upload_project_to_s3(self, project_dir: str, bucket_name: str):
        """Upload project files to S3"""
        logger.info(f"Uploading project to S3 bucket: {bucket_name}")
        
        project_path = Path(project_dir)
        
        for file_path in project_path.rglob('*'):
            if file_path.is_file() and not any(exclude in str(file_path) for exclude in ['.git', '__pycache__', '.pyc']):
                s3_key = f"code/{file_path.relative_to(project_path)}"
                
                try:
                    self.s3.upload_file(str(file_path), bucket_name, s3_key)
                    logger.debug(f"Uploaded: {s3_key}")
                except Exception as e:
                    logger.error(f"Failed to upload {file_path}: {e}")
        
        logger.info("Project upload completed")
    
    def create_cost_alarm(self, threshold: float = 50.0):
        """Create CloudWatch billing alarm"""
        cloudwatch = self.session.client('cloudwatch', region_name='us-east-1')  # Billing metrics are in us-east-1
        
        alarm_name = 'HPE-StyleGAN2-Billing-Alarm'
        
        try:
            cloudwatch.put_metric_alarm(
                AlarmName=alarm_name,
                ComparisonOperator='GreaterThanThreshold',
                EvaluationPeriods=1,
                MetricName='EstimatedCharges',
                Namespace='AWS/Billing',
                Period=86400,  # 24 hours
                Statistic='Maximum',
                Threshold=threshold,
                ActionsEnabled=True,
                AlarmDescription=f'Billing alarm for HPE-StyleGAN2 project - threshold: ${threshold}',
                Dimensions=[
                    {
                        'Name': 'Currency',
                        'Value': 'USD'
                    }
                ],
                Unit='None'
            )
            
            logger.info(f"Billing alarm created with ${threshold} threshold")
            
        except Exception as e:
            logger.error(f"Failed to create billing alarm: {e}")
    
    def get_instance_info(self, instance_id: str) -> Dict:
        """Get information about an instance"""
        response = self.ec2.describe_instances(InstanceIds=[instance_id])
        instance = response['Reservations'][0]['Instances'][0]
        
        return {
            'instance_id': instance_id,
            'state': instance['State']['Name'],
            'instance_type': instance['InstanceType'],
            'public_ip': instance.get('PublicIpAddress'),
            'private_ip': instance.get('PrivateIpAddress'),
            'launch_time': instance['LaunchTime'],
            'key_name': instance.get('KeyName'),
            'security_groups': [sg['GroupName'] for sg in instance['SecurityGroups']]
        }
    
    def terminate_instance(self, instance_id: str):
        """Terminate EC2 instance"""
        logger.info(f"Terminating instance: {instance_id}")
        
        self.ec2.terminate_instances(InstanceIds=[instance_id])
        
        # Wait for termination
        waiter = self.ec2.get_waiter('instance_terminated')
        waiter.wait(InstanceIds=[instance_id])
        
        logger.info(f"Instance {instance_id} terminated successfully")


if __name__ == "__main__":
    # Example usage
    print("AWS Deployment Configuration for HPE-StyleGAN2")
    
    # Initialize deployer
    deployer = AWSDeployer(region='us-east-2')
    
    # Print instance configurations
    print("\nAvailable instance configurations:")
    for config_name, config in AWSConfig.INSTANCE_CONFIGS.items():
        print(f"\n{config_name}:")
        print(f"  Instance Type: {config['instance_type']}")
        print(f"  vCPUs: {config['vcpus']}, Memory: {config['memory_gb']}GB")
        print(f"  GPU: {config['gpu']}")
        print(f"  Cost: ${config['cost_per_hour']:.3f}/hour")
        print(f"  Use Case: {config['use_case']}")
    
    # Example deployment (commented out for safety)
    # instance_id = deployer.launch_instance('development')
    # print(f"Launched instance: {instance_id}")
