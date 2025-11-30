#!/usr/bin/env python3
"""
Phase 5: Closed-Loop Human Feedback - AWS Orchestrator
Manages image generation and human feedback collection on AWS infrastructure.
"""

import os
import boto3
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import uuid

class Phase5AWSOrchestrator:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self._setup_aws_clients()
        self._setup_logging()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        import yaml
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_aws_clients(self):
        """Initialize AWS service clients."""
        self.ec2 = boto3.client('ec2', region_name=self.config['aws']['region'])
        self.s3 = boto3.client('s3', region_name=self.config['aws']['region'])
        self.lambda_client = boto3.client('lambda', region_name=self.config['aws']['region'])
        self.sqs = boto3.client('sqs', region_name=self.config['aws']['region'])
        self.dynamodb = boto3.resource('dynamodb', region_name=self.config['aws']['region'])
        
    def _setup_logging(self):
        """Configure logging for the orchestrator."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def deploy_infrastructure(self) -> Dict[str, str]:
        """Deploy AWS infrastructure for Phase 5."""
        self.logger.info("Deploying Phase 5 AWS infrastructure...")
        
        # Step 1: Create S3 buckets for generated images and feedback
        image_bucket = self._create_s3_bucket('generated-images')
        feedback_bucket = self._create_s3_bucket('human-feedback')
        
        # Step 2: Create DynamoDB tables for feedback tracking
        feedback_table = self._create_dynamodb_table()
        
        # Step 3: Create SQS queues for task management
        generation_queue = self._create_sqs_queue('image-generation-tasks')
        feedback_queue = self._create_sqs_queue('feedback-collection-tasks')
        
        # Step 4: Launch EC2 instances for image generation
        generation_instances = self._launch_generation_instances()
        
        # Step 5: Deploy Lambda functions for feedback processing
        lambda_functions = self._deploy_lambda_functions()
        
        # Step 6: Setup web interface for human feedback
        web_interface = self._setup_web_interface()
        
        infrastructure = {
            'image_bucket': image_bucket,
            'feedback_bucket': feedback_bucket,
            'feedback_table': feedback_table,
            'generation_queue': generation_queue,
            'feedback_queue': feedback_queue,
            'generation_instances': generation_instances,
            'lambda_functions': lambda_functions,
            'web_interface': web_interface
        }
        
        self.logger.info("Infrastructure deployment completed successfully!")
        return infrastructure
    
    def _create_s3_bucket(self, bucket_type: str) -> str:
        """Create S3 bucket with appropriate naming and permissions."""
        bucket_name = f"vlr-{bucket_type}-{uuid.uuid4().hex[:8]}"
        
        try:
            # Create bucket
            if self.config['aws']['region'] == 'us-east-1':
                self.s3.create_bucket(Bucket=bucket_name)
            else:
                self.s3.create_bucket(
                    Bucket=bucket_name,
                    CreateBucketConfiguration={'LocationConstraint': self.config['aws']['region']}
                )
            
            # Enable versioning
            self.s3.put_bucket_versioning(
                Bucket=bucket_name,
                VersioningConfiguration={'Status': 'Enabled'}
            )
            
            # Set CORS for web interface access
            cors_config = {
                'CORSRules': [
                    {
                        'AllowedOrigins': ['*'],
                        'AllowedMethods': ['GET', 'POST', 'PUT'],
                        'AllowedHeaders': ['*'],
                        'MaxAgeSeconds': 3600
                    }
                ]
            }
            self.s3.put_bucket_cors(Bucket=bucket_name, CORSConfiguration=cors_config)
            
            self.logger.info(f"Created S3 bucket: {bucket_name}")
            return bucket_name
            
        except Exception as e:
            self.logger.error(f"Failed to create S3 bucket {bucket_name}: {e}")
            raise
    
    def _create_dynamodb_table(self) -> str:
        """Create DynamoDB table for storing human feedback data."""
        table_name = f"vlr-feedback-{uuid.uuid4().hex[:8]}"
        
        try:
            table = self.dynamodb.create_table(
                TableName=table_name,
                KeySchema=[
                    {'AttributeName': 'feedback_id', 'KeyType': 'HASH'},
                    {'AttributeName': 'timestamp', 'KeyType': 'RANGE'}
                ],
                AttributeDefinitions=[
                    {'AttributeName': 'feedback_id', 'AttributeType': 'S'},
                    {'AttributeName': 'timestamp', 'AttributeType': 'S'},
                    {'AttributeName': 'user_id', 'AttributeType': 'S'},
                    {'AttributeName': 'triplet_id', 'AttributeType': 'S'}
                ],
                BillingMode='PAY_PER_REQUEST',
                GlobalSecondaryIndexes=[
                    {
                        'IndexName': 'UserIndex',
                        'KeySchema': [
                            {'AttributeName': 'user_id', 'KeyType': 'HASH'},
                            {'AttributeName': 'timestamp', 'KeyType': 'RANGE'}
                        ],
                        'Projection': {'ProjectionType': 'ALL'}
                    },
                    {
                        'IndexName': 'TripletIndex',
                        'KeySchema': [
                            {'AttributeName': 'triplet_id', 'KeyType': 'HASH'}
                        ],
                        'Projection': {'ProjectionType': 'ALL'}
                    }
                ],
                Tags=[
                    {'Key': 'Project', 'Value': 'VLR'},
                    {'Key': 'Phase', 'Value': 'Phase5'}
                ]
            )
            
            # Wait for table to be created
            table.wait_until_exists()
            self.logger.info(f"Created DynamoDB table: {table_name}")
            return table_name
            
        except Exception as e:
            self.logger.error(f"Failed to create DynamoDB table: {e}")
            raise
    
    def _create_sqs_queue(self, queue_type: str) -> str:
        """Create SQS queue for task management."""
        queue_name = f"vlr-{queue_type}-{uuid.uuid4().hex[:8]}"
        
        try:
            response = self.sqs.create_queue(
                QueueName=queue_name,
                Attributes={
                    'DelaySeconds': '0',
                    'MaxReceiveCount': '3',
                    'MessageRetentionPeriod': '1209600',  # 14 days
                    'VisibilityTimeoutSeconds': '300'
                }
            )
            
            queue_url = response['QueueUrl']
            self.logger.info(f"Created SQS queue: {queue_name}")
            return queue_url
            
        except Exception as e:
            self.logger.error(f"Failed to create SQS queue {queue_name}: {e}")
            raise
    
    def _launch_generation_instances(self) -> List[str]:
        """Launch EC2 instances for image generation."""
        try:
            # Get latest Deep Learning AMI
            ami_response = self.ec2.describe_images(
                Owners=['amazon'],
                Filters=[
                    {'Name': 'name', 'Values': ['Deep Learning AMI (Ubuntu 20.04) *']},
                    {'Name': 'state', 'Values': ['available']}
                ]
            )
            
            # Sort by creation date and get the latest
            amis = sorted(ami_response['Images'], 
                         key=lambda x: x['CreationDate'], reverse=True)
            ami_id = amis[0]['ImageId']
            
            # Create user data script
            user_data = self._create_user_data_script()
            
            # Launch instances
            instances = []
            for i in range(self.config['phase5']['num_generation_instances']):
                response = self.ec2.run_instances(
                    ImageId=ami_id,
                    MinCount=1,
                    MaxCount=1,
                    InstanceType=self.config['phase5']['generation_instance_type'],
                    KeyName=self.config['aws']['key_pair_name'],
                    SecurityGroups=['default'],
                    UserData=user_data,
                    IamInstanceProfile={'Name': self.config['aws']['instance_profile']},
                    TagSpecifications=[{
                        'ResourceType': 'instance',
                        'Tags': [
                            {'Key': 'Name', 'Value': f'vlr-generation-{i+1}'},
                            {'Key': 'Project', 'Value': 'VLR'},
                            {'Key': 'Phase', 'Value': 'Phase5'},
                            {'Key': 'Role', 'Value': 'ImageGeneration'}
                        ]
                    }]
                )
                
                instance_id = response['Instances'][0]['InstanceId']
                instances.append(instance_id)
                self.logger.info(f"Launched generation instance {i+1}: {instance_id}")
            
            return instances
            
        except Exception as e:
            self.logger.error(f"Failed to launch generation instances: {e}")
            raise
    
    def start_generation_pipeline(self, num_images: int, generation_params: Dict) -> str:
        """Start the image generation pipeline."""
        self.logger.info(f"Starting generation pipeline for {num_images} images...")
        
        # Create generation job
        job_id = f"gen-job-{uuid.uuid4().hex[:8]}"
        
        # Send tasks to generation queue
        generation_tasks = self._create_generation_tasks(num_images, generation_params, job_id)
        for task in generation_tasks:
            self._send_to_queue(self.generation_queue, task)
        
        self.logger.info(f"Created generation job {job_id} with {len(generation_tasks)} tasks")
        return job_id
    
    def start_feedback_collection(self, image_sets: List[List[str]], num_participants: int) -> str:
        """Start human feedback collection for generated image triplets."""
        self.logger.info(f"Starting feedback collection for {len(image_sets)} triplets...")
        
        # Create feedback job
        job_id = f"feedback-job-{uuid.uuid4().hex[:8]}"
        
        # Create triplet comparison tasks
        feedback_tasks = self._create_feedback_tasks(image_sets, num_participants, job_id)
        for task in feedback_tasks:
            self._send_to_queue(self.feedback_queue, task)
        
        self.logger.info(f"Created feedback job {job_id} with {len(feedback_tasks)} tasks")
        return job_id

    def monitor_jobs(self, job_ids: List[str]) -> Dict[str, Dict]:
        """Monitor the status of generation and feedback jobs."""
        status = {}
        
        for job_id in job_ids:
            if job_id.startswith('gen-job'):
                status[job_id] = self._monitor_generation_job(job_id)
            elif job_id.startswith('feedback-job'):
                status[job_id] = self._monitor_feedback_job(job_id)
        
        return status
    
    def cleanup_resources(self, keep_data: bool = True):
        """Clean up AWS resources to avoid charges."""
        self.logger.info("Cleaning up AWS resources...")
        
        # Terminate EC2 instances
        if hasattr(self, 'generation_instances'):
            for instance_id in self.generation_instances:
                try:
                    self.ec2.terminate_instances(InstanceIds=[instance_id])
                    self.logger.info(f"Terminated instance: {instance_id}")
                except Exception as e:
                    self.logger.error(f"Failed to terminate {instance_id}: {e}")
        
        # Delete SQS queues
        for queue_url in [self.generation_queue, self.feedback_queue]:
            try:
                self.sqs.delete_queue(QueueUrl=queue_url)
                self.logger.info(f"Deleted queue: {queue_url}")
            except Exception as e:
                self.logger.error(f"Failed to delete queue {queue_url}: {e}")
        
        if not keep_data:
            # Delete S3 buckets (only if keep_data is False)
            for bucket in [self.image_bucket, self.feedback_bucket]:
                try:
                    self._empty_and_delete_bucket(bucket)
                    self.logger.info(f"Deleted bucket: {bucket}")
                except Exception as e:
                    self.logger.error(f"Failed to delete bucket {bucket}: {e}")
        
        self.logger.info("Cleanup completed!")

if __name__ == "__main__":
    # Example usage
    config_path = "config/phase5_config.yaml"
    orchestrator = Phase5AWSOrchestrator(config_path)
    
    # Deploy infrastructure
    infrastructure = orchestrator.deploy_infrastructure()
    print(f"Infrastructure deployed: {infrastructure}")
    
    # Start generation pipeline
    generation_params = {
        'model_checkpoint': 's3://vlr-models/stylegan2_hpe_final.pth',
        'num_concepts': 100,
        'images_per_concept': 5
    }
    gen_job_id = orchestrator.start_generation_pipeline(500, generation_params)
    
    # Monitor jobs
    status = orchestrator.monitor_jobs([gen_job_id])
    print(f"Job status: {status}")
