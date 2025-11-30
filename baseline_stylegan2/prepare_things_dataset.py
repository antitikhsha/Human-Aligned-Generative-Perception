#!/usr/bin/env python3
"""
THINGS Dataset Preprocessing and Upload Script for AWS S3
Downloads THINGS dataset from OSF and prepares it for StyleGAN2 training
"""

import argparse
import csv
import json
import logging
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import boto3
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

class THINGSDatasetProcessor:
    """Processes and uploads THINGS dataset to AWS S3"""
    
    def __init__(self, output_dir: str, s3_bucket: str, aws_region: str = 'us-east-2'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.s3_bucket = s3_bucket
        self.aws_region = aws_region
        self.s3_client = boto3.client('s3', region_name=aws_region)
        
        # THINGS dataset URLs (from OSF)
        self.urls = {
            'images': 'https://osf.io/jum2f/download',  # Main image archive
            'metadata': 'https://osf.io/8xuk3/download',  # Concept metadata
            'embeddings': 'https://osf.io/3zx5c/download',  # Human similarity embeddings
            'triplets': 'https://osf.io/jk2es/download'  # Triplet similarity data
        }
        
    def download_file(self, url: str, output_path: Path, description: str = "Downloading"):
        """Download file with progress bar"""
        def progress_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, (downloaded * 100) // total_size)
                progress_bar.update(min(block_size, total_size - downloaded))
        
        with tqdm(total=None, unit='B', unit_scale=True, desc=description) as progress_bar:
            urllib.request.urlretrieve(url, output_path, progress_hook)
        
        logging.info(f"Downloaded {description} to {output_path}")
    
    def download_dataset(self):
        """Download all THINGS dataset components"""
        logging.info("Downloading THINGS dataset components...")
        
        downloads_dir = self.output_dir / 'downloads'
        downloads_dir.mkdir(exist_ok=True)
        
        for component, url in self.urls.items():
            output_path = downloads_dir / f"{component}.zip"
            if not output_path.exists():
                self.download_file(url, output_path, f"THINGS {component}")
            else:
                logging.info(f"{component} already downloaded")
    
    def extract_archives(self):
        """Extract downloaded archives"""
        logging.info("Extracting downloaded archives...")
        
        downloads_dir = self.output_dir / 'downloads'
        extracted_dir = self.output_dir / 'extracted'
        extracted_dir.mkdir(exist_ok=True)
        
        for component in self.urls.keys():
            archive_path = downloads_dir / f"{component}.zip"
            extract_path = extracted_dir / component
            
            if archive_path.exists() and not extract_path.exists():
                logging.info(f"Extracting {component}...")
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_path)
                logging.info(f"Extracted {component} to {extract_path}")
            else:
                logging.info(f"{component} already extracted or archive not found")
    
    def process_metadata(self) -> Dict:
        """Process concept metadata and create mappings"""
        logging.info("Processing concept metadata...")
        
        extracted_dir = self.output_dir / 'extracted'
        metadata_dir = extracted_dir / 'metadata'
        
        # Find metadata files
        concept_files = list(metadata_dir.glob('**/*concept*.csv'))
        if not concept_files:
            raise FileNotFoundError("Concept metadata files not found")
        
        # Load concept information
        concepts = {}
        for concept_file in concept_files:
            with open(concept_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    concept_id = row.get('uniqueID', row.get('concept_id'))
                    concept_name = row.get('Word', row.get('concept_name'))
                    if concept_id and concept_name:
                        concepts[concept_id] = {
                            'name': concept_name,
                            'category': row.get('category', 'unknown'),
                            'images': []
                        }
        
        logging.info(f"Loaded {len(concepts)} concepts")
        return concepts
    
    def process_embeddings(self) -> Dict:
        """Process human perceptual embeddings"""
        logging.info("Processing human perceptual embeddings...")
        
        extracted_dir = self.output_dir / 'extracted'
        embeddings_dir = extracted_dir / 'embeddings'
        
        # Find embedding files
        embedding_files = list(embeddings_dir.glob('**/*.npy')) + list(embeddings_dir.glob('**/*.csv'))
        
        embeddings = {}
        for emb_file in embedding_files:
            try:
                if emb_file.suffix == '.npy':
                    data = np.load(emb_file)
                    # Assuming the embedding file contains concept IDs and embeddings
                    # Format may vary, adjust accordingly
                    if data.ndim == 2:  # [num_concepts, embedding_dim]
                        for i, embedding in enumerate(data):
                            concept_id = f"concept_{i:04d}"  # Adjust based on actual format
                            embeddings[concept_id] = embedding.tolist()
                elif emb_file.suffix == '.csv':
                    df = pd.read_csv(emb_file)
                    # Process CSV format embeddings
                    for _, row in df.iterrows():
                        concept_id = row.get('uniqueID', row.get('concept_id'))
                        if concept_id:
                            # Extract embedding dimensions (adjust column names as needed)
                            embedding_cols = [col for col in df.columns if col.startswith('dim_') or col.startswith('embedding_')]
                            if embedding_cols:
                                embedding = [float(row[col]) for col in embedding_cols]
                                embeddings[concept_id] = embedding
            except Exception as e:
                logging.warning(f"Failed to process embedding file {emb_file}: {e}")
        
        logging.info(f"Processed embeddings for {len(embeddings)} concepts")
        return embeddings
    
    def organize_images(self, concepts: Dict) -> Dict:
        """Organize images by concept and prepare for training"""
        logging.info("Organizing images by concept...")
        
        extracted_dir = self.output_dir / 'extracted'
        images_dir = extracted_dir / 'images'
        
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(images_dir.glob(f'**/*{ext}'))
            image_files.extend(images_dir.glob(f'**/*{ext.upper()}'))
        
        logging.info(f"Found {len(image_files)} image files")
        
        # Organize images by concept
        processed_dir = self.output_dir / 'processed'
        processed_dir.mkdir(exist_ok=True)
        
        organized_concepts = {}
        unmatched_images = []
        
        for image_path in tqdm(image_files, desc="Organizing images"):
            # Extract concept ID from filename or directory structure
            # This depends on the actual THINGS dataset structure
            concept_id = self.extract_concept_id(image_path)
            
            if concept_id and concept_id in concepts:
                # Create concept directory
                concept_dir = processed_dir / concept_id
                concept_dir.mkdir(exist_ok=True)
                
                # Process and save image
                processed_image_path = concept_dir / f"{image_path.stem}.jpg"
                
                try:
                    # Load, process, and save image
                    with Image.open(image_path) as img:
                        # Convert to RGB if necessary
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        
                        # Resize to consistent size (optional)
                        img = img.resize((256, 256), Image.Resampling.LANCZOS)
                        
                        # Save processed image
                        img.save(processed_image_path, 'JPEG', quality=95)
                    
                    # Track in concept
                    if concept_id not in organized_concepts:
                        organized_concepts[concept_id] = concepts[concept_id].copy()
                        organized_concepts[concept_id]['images'] = []
                    
                    organized_concepts[concept_id]['images'].append(str(processed_image_path.relative_to(processed_dir)))
                    
                except Exception as e:
                    logging.warning(f"Failed to process image {image_path}: {e}")
                    
            else:
                unmatched_images.append(str(image_path))
        
        logging.info(f"Organized {len(organized_concepts)} concepts")
        logging.info(f"Unmatched images: {len(unmatched_images)}")
        
        return organized_concepts
    
    def extract_concept_id(self, image_path: Path) -> str:
        """Extract concept ID from image path"""
        # This function needs to be adapted based on actual THINGS dataset structure
        # Common patterns:
        
        # Pattern 1: concept ID in filename
        filename = image_path.stem
        if '_' in filename:
            parts = filename.split('_')
            # Look for concept ID pattern
            for part in parts:
                if part.isdigit() and len(part) >= 3:
                    return f"concept_{part}"
        
        # Pattern 2: concept ID in directory name
        for parent in image_path.parents:
            parent_name = parent.name
            if parent_name.startswith('concept_') or parent_name.isdigit():
                return parent_name if parent_name.startswith('concept_') else f"concept_{parent_name}"
        
        # Pattern 3: Extract from full path
        path_str = str(image_path)
        # Add more extraction logic based on actual dataset structure
        
        return None
    
    def create_dataset_manifest(self, concepts: Dict, embeddings: Dict):
        """Create dataset manifest for training"""
        logging.info("Creating dataset manifest...")
        
        manifest = {
            'dataset': 'THINGS',
            'version': '1.0',
            'total_concepts': len(concepts),
            'total_images': sum(len(concept['images']) for concept in concepts.values()),
            'has_embeddings': len(embeddings) > 0,
            'embedding_dim': len(next(iter(embeddings.values()))) if embeddings else 0,
            'concepts': concepts,
            'embeddings': embeddings
        }
        
        manifest_path = self.output_dir / 'processed' / 'dataset_manifest.json'
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logging.info(f"Dataset manifest saved to {manifest_path}")
        return manifest
    
    def upload_to_s3(self):
        """Upload processed dataset to S3"""
        logging.info(f"Uploading dataset to S3 bucket: {self.s3_bucket}")
        
        processed_dir = self.output_dir / 'processed'
        
        # Get list of all files to upload
        files_to_upload = []
        for file_path in processed_dir.rglob('*'):
            if file_path.is_file():
                relative_path = file_path.relative_to(processed_dir)
                s3_key = f"things_dataset/{relative_path}"
                files_to_upload.append((file_path, s3_key))
        
        logging.info(f"Uploading {len(files_to_upload)} files to S3...")
        
        # Upload with progress bar
        for file_path, s3_key in tqdm(files_to_upload, desc="Uploading to S3"):
            try:
                self.s3_client.upload_file(
                    str(file_path),
                    self.s3_bucket,
                    s3_key
                )
            except Exception as e:
                logging.error(f"Failed to upload {file_path}: {e}")
        
        logging.info("Dataset upload completed")
    
    def create_s3_bucket_if_not_exists(self):
        """Create S3 bucket if it doesn't exist"""
        try:
            # Check if bucket exists
            self.s3_client.head_bucket(Bucket=self.s3_bucket)
            logging.info(f"S3 bucket '{self.s3_bucket}' already exists")
        except:
            try:
                # Create bucket
                if self.aws_region == 'us-east-1':
                    self.s3_client.create_bucket(Bucket=self.s3_bucket)
                else:
                    self.s3_client.create_bucket(
                        Bucket=self.s3_bucket,
                        CreateBucketConfiguration={'LocationConstraint': self.aws_region}
                    )
                logging.info(f"Created S3 bucket: {self.s3_bucket}")
            except Exception as e:
                logging.error(f"Failed to create S3 bucket: {e}")
                raise
    
    def process_dataset(self, skip_download: bool = False, skip_upload: bool = False):
        """Complete dataset processing pipeline"""
        logging.info("Starting THINGS dataset processing...")
        
        try:
            if not skip_download:
                self.download_dataset()
                self.extract_archives()
            
            # Process components
            concepts = self.process_metadata()
            embeddings = self.process_embeddings()
            organized_concepts = self.organize_images(concepts)
            
            # Create manifest
            manifest = self.create_dataset_manifest(organized_concepts, embeddings)
            
            if not skip_upload:
                self.create_s3_bucket_if_not_exists()
                self.upload_to_s3()
            
            logging.info("Dataset processing completed successfully!")
            
            # Print summary
            print("\n" + "="*50)
            print("DATASET PROCESSING SUMMARY")
            print("="*50)
            print(f"Total concepts: {manifest['total_concepts']}")
            print(f"Total images: {manifest['total_images']}")
            print(f"Has embeddings: {manifest['has_embeddings']}")
            print(f"Embedding dimension: {manifest['embedding_dim']}")
            print(f"S3 bucket: {self.s3_bucket}")
            print("="*50)
            
        except Exception as e:
            logging.error(f"Dataset processing failed: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description='Process THINGS dataset for AWS StyleGAN2 training')
    parser.add_argument('--output-dir', type=str, required=True, help='Local output directory')
    parser.add_argument('--s3-bucket', type=str, required=True, help='S3 bucket name for dataset storage')
    parser.add_argument('--aws-region', type=str, default='us-east-2', help='AWS region')
    parser.add_argument('--skip-download', action='store_true', help='Skip download step')
    parser.add_argument('--skip-upload', action='store_true', help='Skip S3 upload step')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Process dataset
    processor = THINGSDatasetProcessor(args.output_dir, args.s3_bucket, args.aws_region)
    processor.process_dataset(args.skip_download, args.skip_upload)

if __name__ == '__main__':
    main()
