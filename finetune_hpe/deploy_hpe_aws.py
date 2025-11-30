#!/usr/bin/env python3
"""
AWS EC2 Deployment Script for HPE Fine-tuning
Step 5.3: Fine-tune HPE with New Data

This script automates the deployment and execution of HPE fine-tuning on AWS EC2.
"""

import boto3
import time
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from botocore.exceptions import ClientError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EC2HPEDeployer:
    """Deploy and manage HPE fine-tuning on EC2"""
    
    def __init__(self, region: str = 'us-east-2'):
        self.region = region
        self.ec2 = boto3.client('ec2', region_name=region)
        self.ec2_resource = boto3.resource('ec2', region_name=region)
        self.s3 = boto3.client('s3')
        
    def create_security_group(self, name: str = 'hpe-training-sg') -> str:
        """Create security group for training instances"""
        try:
            response = self.ec2.create_security_group(
                GroupName=name,
                Description='Security group for HPE training instances'
            )
            sg_id = response['GroupId']
            
            # Add SSH rule
            self.ec2.authorize_security_group_ingress(
                GroupId=sg_id,
                IpPermissions=[
                    {
                        'IpProtocol': 'tcp',
                        'FromPort': 22,
                        'ToPort': 22,
                        'IpRanges': [{'CidrIp': '0.0.0.0/0'}]
                    }
                ]
            )
            
            logger.info(f"Created security group: {sg_id}")
            return sg_id
            
        except ClientError as e:
            if 'InvalidGroup.Duplicate' in str(e):
                # Security group already exists
                groups = self.ec2.describe_security_groups(GroupNames=[name])
                sg_id = groups['SecurityGroups'][0]['GroupId']
                logger.info(f"Using existing security group: {sg_id}")
                return sg_id
            else:
                logger.error(f"Failed to create security group: {e}")
                raise
    
    def get_latest_dlami(self, instance_type: str) -> str:
        """Get the latest Deep Learning AMI for the instance type"""
        # Search for latest Amazon Deep Learning AMI
        images = self.ec2.describe_images(
            Owners=['amazon'],
            Filters=[
                {'Name': 'name', 'Values': ['Deep Learning AMI (Amazon Linux 2)*']},
                {'Name': 'state', 'Values': ['available']},
                {'Name': 'architecture', 'Values': ['x86_64']}
            ]
        )
        
        if not images['Images']:
            raise RuntimeError("No suitable Deep Learning AMI found")
        
        # Sort by creation date and get the latest
        latest_ami = sorted(
            images['Images'],
            key=lambda x: x['CreationDate'],
            reverse=True
        )[0]
        
        logger.info(f"Selected AMI: {latest_ami['ImageId']} - {latest_ami['Name']}")
        return latest_ami['ImageId']
    
    def create_user_data_script(self, config: Dict) -> str:
        """Create user data script for instance initialization"""
        s3_bucket = config['output']['s3_bucket']
        s3_prefix = config['output'].get('s3_prefix', 'hpe_training')
        
        script = f"""#!/bin/bash
# HPE Fine-tuning Instance Setup Script

set -e
cd /home/ec2-user

# Update system
sudo yum update -y

# Install additional packages
sudo yum install -y git htop tree

# Setup conda environment
source /opt/miniconda3/bin/activate
conda create -n hpe python=3.8 -y
conda activate hpe

# Install PyTorch and dependencies
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install boto3 wandb pyyaml tqdm pillow pandas

# Create working directory
mkdir -p hpe_training
cd hpe_training

# Download training script and config from S3
aws s3 cp s3://{s3_bucket}/{s3_prefix}/scripts/hpe_finetune_aws.py ./
aws s3 cp s3://{s3_bucket}/{s3_prefix}/configs/hpe_finetune_config.yaml ./

# Create training script
cat > run_training.sh << 'EOF'
#!/bin/bash
source /opt/miniconda3/bin/activate hpe

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export WANDB_API_KEY=${{WANDB_API_KEY:-""}}

# Run training
python hpe_finetune_aws.py --config hpe_finetune_config.yaml

# Save final model to S3
aws s3 sync checkpoints/ s3://{s3_bucket}/{s3_prefix}/final_checkpoints/

# Instance auto-shutdown (optional safety measure)
echo "Training completed. Shutting down instance in 10 minutes..."
sudo shutdown -h +10
EOF

chmod +x run_training.sh

# Setup auto-start training (after instance is fully initialized)
cat > start_training.sh << 'EOF'
#!/bin/bash
# Wait for full initialization
sleep 60

# Start training in screen session
screen -dmS hpe_training ./run_training.sh

# Create status file
echo "Training started at $(date)" > training_status.txt
EOF

chmod +x start_training.sh

# Schedule training to start after initialization
echo "/home/ec2-user/hpe_training/start_training.sh" | sudo tee -a /etc/rc.local
sudo chmod +x /etc/rc.local

# Create monitoring script
cat > monitor_training.sh << 'EOF'
#!/bin/bash
echo "=== HPE Training Status ==="
echo "Instance: $(curl -s http://169.254.169.254/latest/meta-data/instance-id)"
echo "Status file:"
cat training_status.txt 2>/dev/null || echo "Status file not found"
echo
echo "=== Screen sessions ==="
screen -list
echo
echo "=== GPU usage ==="
nvidia-smi
echo
echo "=== Disk usage ==="
df -h
echo
echo "=== Recent training logs ==="
tail -20 nohup.out 2>/dev/null || echo "No training logs found"
EOF

chmod +x monitor_training.sh

echo "HPE training instance setup completed!"
"""
        return script
    
    def launch_training_instance(
        self,
        instance_type: str,
        config: Dict,
        key_pair_name: str,
        spot_instance: bool = True,
        max_price: Optional[str] = None
    ) -> str:
        """Launch EC2 instance for HPE training"""
        
        # Get AMI and security group
        ami_id = self.get_latest_dlami(instance_type)
        sg_id = self.create_security_group()
        
        # Create user data script
        user_data = self.create_user_data_script(config)
        
        # Instance configuration
        instance_config = {
            'ImageId': ami_id,
            'InstanceType': instance_type,
            'KeyName': key_pair_name,
            'SecurityGroupIds': [sg_id],
            'UserData': user_data,
            'IamInstanceProfile': {
                'Name': 'EC2-S3-Access'  # Assumes this role exists
            },
            'BlockDeviceMappings': [
                {
                    'DeviceName': '/dev/xvda',
                    'Ebs': {
                        'VolumeSize': 100,  # 100GB root volume
                        'VolumeType': 'gp3',
                        'DeleteOnTermination': True
                    }
                }
            ],
            'TagSpecifications': [
                {
                    'ResourceType': 'instance',
                    'Tags': [
                        {'Key': 'Name', 'Value': 'HPE-Training-Instance'},
                        {'Key': 'Project', 'Value': 'VLR-HPE-Research'},
                        {'Key': 'AutoShutdown', 'Value': 'true'}
                    ]
                }
            ]
        }
        
        try:
            if spot_instance:
                # Launch spot instance
                if not max_price:
                    max_price = self.get_spot_price_suggestion(instance_type)
                
                response = self.ec2.request_spot_instances(
                    SpotPrice=max_price,
                    InstanceCount=1,
                    LaunchSpecification=instance_config
                )
                spot_request_id = response['SpotInstanceRequests'][0]['SpotInstanceRequestId']
                
                logger.info(f"Submitted spot instance request: {spot_request_id}")
                logger.info(f"Waiting for spot instance to launch...")
                
                # Wait for spot request to be fulfilled
                instance_id = self.wait_for_spot_instance(spot_request_id)
                
            else:
                # Launch on-demand instance
                response = self.ec2.run_instances(
                    MinCount=1,
                    MaxCount=1,
                    **instance_config
                )
                instance_id = response['Instances'][0]['InstanceId']
            
            logger.info(f"Launched instance: {instance_id}")
            
            # Wait for instance to be running
            waiter = self.ec2.get_waiter('instance_running')
            waiter.wait(InstanceIds=[instance_id])
            
            # Get instance details
            instance = self.ec2.describe_instances(InstanceIds=[instance_id])
            public_ip = instance['Reservations'][0]['Instances'][0].get('PublicIpAddress')
            
            logger.info(f"Instance is running. Public IP: {public_ip}")
            
            return instance_id
            
        except Exception as e:
            logger.error(f"Failed to launch instance: {e}")
            raise
    
    def wait_for_spot_instance(self, spot_request_id: str, timeout: int = 600) -> str:
        """Wait for spot instance request to be fulfilled"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            response = self.ec2.describe_spot_instance_requests(
                SpotInstanceRequestIds=[spot_request_id]
            )
            
            request = response['SpotInstanceRequests'][0]
            state = request['State']
            
            if state == 'active':
                return request['InstanceId']
            elif state == 'failed':
                raise RuntimeError(f"Spot request failed: {request.get('Fault', {}).get('Message')}")
            
            logger.info(f"Spot request state: {state}")
            time.sleep(30)
        
        raise RuntimeError("Timeout waiting for spot instance")
    
    def get_spot_price_suggestion(self, instance_type: str) -> str:
        """Get suggested spot price based on recent pricing"""
        try:
            response = self.ec2.describe_spot_price_history(
                InstanceTypes=[instance_type],
                ProductDescriptions=['Linux/UNIX'],
                MaxResults=10
            )
            
            if response['SpotPrices']:
                recent_price = float(response['SpotPrices'][0]['SpotPrice'])
                suggested_price = recent_price * 1.2  # 20% buffer
                
                logger.info(f"Recent spot price: ${recent_price:.4f}/hour")
                logger.info(f"Suggested max price: ${suggested_price:.4f}/hour")
                
                return f"{suggested_price:.4f}"
            else:
                # Fallback prices
                fallback_prices = {
                    'g4dn.xlarge': '0.60',
                    'p3.2xlarge': '3.50',
                    'p4d.24xlarge': '30.00'
                }
                return fallback_prices.get(instance_type, '1.00')
                
        except Exception as e:
            logger.warning(f"Could not get spot pricing: {e}")
            return '1.00'
    
    def upload_training_files(self, bucket: str, prefix: str):
        """Upload training scripts and configs to S3"""
        files_to_upload = [
            ('hpe_finetune_aws.py', f'{prefix}/scripts/hpe_finetune_aws.py'),
            ('hpe_finetune_config.yaml', f'{prefix}/configs/hpe_finetune_config.yaml')
        ]
        
        for local_file, s3_key in files_to_upload:
            if Path(local_file).exists():
                self.s3.upload_file(local_file, bucket, s3_key)
                logger.info(f"Uploaded {local_file} to s3://{bucket}/{s3_key}")
    
    def monitor_instance(self, instance_id: str):
        """Display instance status and training progress"""
        try:
            response = self.ec2.describe_instances(InstanceIds=[instance_id])
            instance = response['Reservations'][0]['Instances'][0]
            
            print(f"\n=== Instance Status ===")
            print(f"Instance ID: {instance_id}")
            print(f"State: {instance['State']['Name']}")
            print(f"Type: {instance['InstanceType']}")
            print(f"Public IP: {instance.get('PublicIpAddress', 'N/A')}")
            print(f"Launch Time: {instance['LaunchTime']}")
            
            # Show estimated costs
            uptime_hours = (time.time() - instance['LaunchTime'].timestamp()) / 3600
            print(f"Uptime: {uptime_hours:.2f} hours")
            
        except Exception as e:
            logger.error(f"Error monitoring instance: {e}")
    
    def terminate_instance(self, instance_id: str):
        """Terminate training instance"""
        try:
            self.ec2.terminate_instances(InstanceIds=[instance_id])
            logger.info(f"Terminating instance: {instance_id}")
            
            # Wait for termination
            waiter = self.ec2.get_waiter('instance_terminated')
            waiter.wait(InstanceIds=[instance_id])
            
            logger.info("Instance terminated successfully")
            
        except Exception as e:
            logger.error(f"Error terminating instance: {e}")

def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(description="Deploy HPE fine-tuning on AWS EC2")
    parser.add_argument('--config', required=True, help='HPE training config file')
    parser.add_argument('--instance-type', default='p3.2xlarge', help='EC2 instance type')
    parser.add_argument('--key-pair', required=True, help='EC2 key pair name')
    parser.add_argument('--region', default='us-east-2', help='AWS region')
    parser.add_argument('--spot', action='store_true', help='Use spot instances')
    parser.add_argument('--max-price', help='Max spot price')
    parser.add_argument('--monitor', help='Monitor existing instance ID')
    parser.add_argument('--terminate', help='Terminate instance ID')
    
    args = parser.parse_args()
    
    deployer = EC2HPEDeployer(region=args.region)
    
    try:
        if args.monitor:
            deployer.monitor_instance(args.monitor)
            return 0
        
        if args.terminate:
            deployer.terminate_instance(args.terminate)
            return 0
        
        # Load config
        with open(args.config, 'r') as f:
            import yaml
            config = yaml.safe_load(f)
        
        # Upload training files
        bucket = config['output']['s3_bucket']
        prefix = config['output'].get('s3_prefix', 'hpe_training')
        deployer.upload_training_files(bucket, prefix)
        
        # Launch training instance
        instance_id = deployer.launch_training_instance(
            instance_type=args.instance_type,
            config=config,
            key_pair_name=args.key_pair,
            spot_instance=args.spot,
            max_price=args.max_price
        )
        
        print(f"\n✅ Training instance launched successfully!")
        print(f"Instance ID: {instance_id}")
        print(f"Instance Type: {args.instance_type}")
        print(f"Region: {args.region}")
        print(f"\nTo monitor progress:")
        print(f"python deploy_hpe_aws.py --monitor {instance_id}")
        print(f"\nTo terminate when done:")
        print(f"python deploy_hpe_aws.py --terminate {instance_id}")
        print(f"\nSSH access:")
        print(f"ssh -i ~/.ssh/{args.key_pair}.pem ec2-user@<public-ip>")
        
        return 0
        
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
