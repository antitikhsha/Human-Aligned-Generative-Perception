#!/usr/bin/env python3
"""
Step 5.1: Generate New Images - AWS Worker
Handles image generation tasks from SQS queue using trained StyleGAN2+HPE model.
"""

import os
import json
import boto3
import torch
import numpy as np
import logging
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import uuid
from io import BytesIO
from PIL import Image
import pickle

class ImageGenerationWorker:
    def __init__(self):
        self._setup_logging()
        self._setup_aws_clients()
        self._setup_model()
        
    def _setup_logging(self):
        """Configure logging for the worker."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _setup_aws_clients(self):
        """Initialize AWS service clients."""
        self.s3 = boto3.client('s3')
        self.sqs = boto3.client('sqs')
        
        # Get queue URLs from environment variables
        self.generation_queue_url = os.environ['GENERATION_QUEUE_URL']
        self.image_bucket = os.environ['IMAGE_BUCKET']
        self.model_bucket = os.environ['MODEL_BUCKET']
        
    def _setup_model(self):
        """Load the trained StyleGAN2+HPE model."""
        self.logger.info("Loading trained model from S3...")
        
        # Download model checkpoint
        model_key = os.environ['MODEL_CHECKPOINT_KEY']
        local_model_path = '/tmp/stylegan2_hpe_model.pth'
        
        try:
            self.s3.download_file(self.model_bucket, model_key, local_model_path)
            
            # Load model
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = torch.load(local_model_path, map_location=self.device)
            self.model.eval()
            
            # Load HPE embeddings
            hpe_key = os.environ.get('HPE_EMBEDDINGS_KEY', 'embeddings/hpe_embeddings.pkl')
            hpe_local_path = '/tmp/hpe_embeddings.pkl'
            self.s3.download_file(self.model_bucket, hpe_key, hpe_local_path)
            
            with open(hpe_local_path, 'rb') as f:
                self.hpe_embeddings = pickle.load(f)
            
            self.logger.info("Model loaded successfully!")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def run(self):
        """Main worker loop - processes generation tasks from SQS."""
        self.logger.info("Starting image generation worker...")
        
        while True:
            try:
                # Poll for messages
                response = self.sqs.receive_message(
                    QueueUrl=self.generation_queue_url,
                    MaxNumberOfMessages=1,
                    WaitTimeSeconds=20,  # Long polling
                    VisibilityTimeoutSeconds=300
                )
                
                if 'Messages' not in response:
                    self.logger.info("No messages in queue, continuing to poll...")
                    continue
                
                message = response['Messages'][0]
                receipt_handle = message['ReceiptHandle']
                
                try:
                    # Parse task
                    task = json.loads(message['Body'])
                    self.logger.info(f"Processing task: {task['task_id']}")
                    
                    # Generate images
                    generated_images = self._process_generation_task(task)
                    
                    # Upload results to S3
                    self._upload_generated_images(task, generated_images)
                    
                    # Delete message from queue
                    self.sqs.delete_message(
                        QueueUrl=self.generation_queue_url,
                        ReceiptHandle=receipt_handle
                    )
                    
                    self.logger.info(f"Successfully completed task: {task['task_id']}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to process message: {e}")
                    # Message will become visible again for retry
                    
            except KeyboardInterrupt:
                self.logger.info("Received interrupt signal, shutting down...")
                break
            except Exception as e:
                self.logger.error(f"Worker error: {e}")
                time.sleep(5)  # Wait before retrying
    
    def _process_generation_task(self, task: Dict) -> List[Dict]:
        """Process a single generation task."""
        task_type = task['task_type']
        
        if task_type == 'concept_generation':
            return self._generate_concept_variations(task)
        elif task_type == 'interpolation_generation':
            return self._generate_interpolations(task)
        elif task_type == 'hpe_guided_generation':
            return self._generate_hpe_guided_images(task)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    def _generate_concept_variations(self, task: Dict) -> List[Dict]:
        """Generate variations of a specific concept."""
        concept_name = task['concept_name']
        num_variations = task['num_variations']
        
        self.logger.info(f"Generating {num_variations} variations of concept: {concept_name}")
        
        # Get concept's HPE embedding
        if concept_name not in self.hpe_embeddings:
            raise ValueError(f"Concept {concept_name} not found in HPE embeddings")
        
        hpe_embedding = self.hpe_embeddings[concept_name]
        
        generated_images = []
        
        for i in range(num_variations):
            # Sample random noise
            z = torch.randn(1, 512, device=self.device)
            
            # Add noise to HPE embedding for variation
            noise_scale = task.get('variation_scale', 0.1)
            hpe_noise = torch.randn_like(torch.tensor(hpe_embedding)) * noise_scale
            perturbed_hpe = torch.tensor(hpe_embedding + hpe_noise, device=self.device)
            
            # Generate image
            with torch.no_grad():
                generated_img = self.model.generate_with_hpe(z, perturbed_hpe.unsqueeze(0))
                generated_img = self._postprocess_image(generated_img)
            
            image_data = {
                'image': generated_img,
                'concept_name': concept_name,
                'variation_id': i,
                'hpe_embedding': hpe_embedding.tolist(),
                'perturbed_hpe': perturbed_hpe.cpu().numpy().tolist(),
                'noise_vector': z.cpu().numpy().tolist(),
                'generation_params': {
                    'variation_scale': noise_scale,
                    'model_checkpoint': task.get('model_checkpoint')
                }
            }
            
            generated_images.append(image_data)
        
        return generated_images
    
    def _generate_interpolations(self, task: Dict) -> List[Dict]:
        """Generate interpolations between two concepts."""
        concept_a = task['concept_a']
        concept_b = task['concept_b']
        num_steps = task['num_steps']
        
        self.logger.info(f"Generating {num_steps} interpolations between {concept_a} and {concept_b}")
        
        # Get HPE embeddings for both concepts
        hpe_a = torch.tensor(self.hpe_embeddings[concept_a], device=self.device)
        hpe_b = torch.tensor(self.hpe_embeddings[concept_b], device=self.device)
        
        generated_images = []
        
        for i in range(num_steps):
            # Interpolation weight
            alpha = i / (num_steps - 1)
            
            # Interpolate HPE embeddings
            interpolated_hpe = (1 - alpha) * hpe_a + alpha * hpe_b
            
            # Sample noise
            z = torch.randn(1, 512, device=self.device)
            
            # Generate image
            with torch.no_grad():
                generated_img = self.model.generate_with_hpe(z, interpolated_hpe.unsqueeze(0))
                generated_img = self._postprocess_image(generated_img)
            
            image_data = {
                'image': generated_img,
                'concept_a': concept_a,
                'concept_b': concept_b,
                'interpolation_step': i,
                'alpha': alpha,
                'interpolated_hpe': interpolated_hpe.cpu().numpy().tolist(),
                'noise_vector': z.cpu().numpy().tolist(),
                'generation_params': task.get('generation_params', {})
            }
            
            generated_images.append(image_data)
        
        return generated_images
    
    def _generate_hpe_guided_images(self, task: Dict) -> List[Dict]:
        """Generate images guided by specific HPE coordinates."""
        target_hpe = torch.tensor(task['target_hpe'], device=self.device)
        num_samples = task['num_samples']
        
        self.logger.info(f"Generating {num_samples} images for target HPE coordinates")
        
        generated_images = []
        
        for i in range(num_samples):
            # Sample different noise vectors for variety
            z = torch.randn(1, 512, device=self.device)
            
            # Generate image
            with torch.no_grad():
                generated_img = self.model.generate_with_hpe(z, target_hpe.unsqueeze(0))
                generated_img = self._postprocess_image(generated_img)
            
            image_data = {
                'image': generated_img,
                'target_hpe': target_hpe.cpu().numpy().tolist(),
                'sample_id': i,
                'noise_vector': z.cpu().numpy().tolist(),
                'generation_params': task.get('generation_params', {})
            }
            
            generated_images.append(image_data)
        
        return generated_images
    
    def _postprocess_image(self, img_tensor: torch.Tensor) -> Image.Image:
        """Convert model output tensor to PIL Image."""
        # Assuming img_tensor is in [-1, 1] range
        img_np = ((img_tensor.squeeze(0).cpu().numpy() + 1) * 127.5).astype(np.uint8)
        
        # Convert CHW to HWC
        if img_np.ndim == 3:
            img_np = np.transpose(img_np, (1, 2, 0))
        
        return Image.fromarray(img_np)
    
    def _upload_generated_images(self, task: Dict, generated_images: List[Dict]):
        """Upload generated images and metadata to S3."""
        job_id = task['job_id']
        task_id = task['task_id']
        
        # Create task folder structure
        base_key = f"generated_images/{job_id}/{task_id}"
        
        for i, img_data in enumerate(generated_images):
            # Save image
            img_key = f"{base_key}/image_{i:04d}.png"
            img_buffer = BytesIO()
            img_data['image'].save(img_buffer, format='PNG')
            img_buffer.seek(0)
            
            self.s3.upload_fileobj(
                img_buffer,
                self.image_bucket,
                img_key,
                ExtraArgs={'ContentType': 'image/png'}
            )
            
            # Save metadata
            metadata = {k: v for k, v in img_data.items() if k != 'image'}
            metadata['s3_key'] = img_key
            metadata['upload_timestamp'] = datetime.utcnow().isoformat()
            
            metadata_key = f"{base_key}/metadata_{i:04d}.json"
            self.s3.put_object(
                Bucket=self.image_bucket,
                Key=metadata_key,
                Body=json.dumps(metadata, indent=2),
                ContentType='application/json'
            )
        
        # Create task summary
        task_summary = {
            'task_id': task_id,
            'job_id': job_id,
            'task_type': task['task_type'],
            'num_images_generated': len(generated_images),
            'completion_timestamp': datetime.utcnow().isoformat(),
            'base_s3_key': base_key
        }
        
        summary_key = f"{base_key}/task_summary.json"
        self.s3.put_object(
            Bucket=self.image_bucket,
            Key=summary_key,
            Body=json.dumps(task_summary, indent=2),
            ContentType='application/json'
        )
        
        self.logger.info(f"Uploaded {len(generated_images)} images to S3: {base_key}")

def main():
    """Run the image generation worker."""
    worker = ImageGenerationWorker()
    worker.run()

if __name__ == "__main__":
    main()
