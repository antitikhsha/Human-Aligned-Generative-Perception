"""
THINGS Dataset Handler for HPE-StyleGAN2 Training
Processes THINGS dataset with object images and triplet similarity judgments
for human-aligned generative modeling.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets.utils import download_url
import numpy as np
import pandas as pd
from PIL import Image
import json
import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import requests
import zipfile
from tqdm import tqdm
import pickle

logger = logging.getLogger(__name__)


class THINGSDataset(Dataset):
    """
    THINGS Dataset for HPE-StyleGAN2 training
    
    Provides object images with corresponding human perceptual embeddings
    and triplet similarity judgments.
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        image_size: int = 256,
        download: bool = True,
        load_triplets: bool = True,
        max_triplets_per_concept: int = 100
    ):
        """
        Initialize THINGS dataset
        
        Args:
            data_root: Root directory for dataset
            split: Dataset split ('train', 'val', 'test')
            image_size: Target image size for training
            download: Whether to download dataset if not found
            load_triplets: Whether to load triplet similarity data
            max_triplets_per_concept: Maximum triplets per concept
        """
        self.data_root = Path(data_root)
        self.split = split
        self.image_size = image_size
        self.load_triplets = load_triplets
        self.max_triplets_per_concept = max_triplets_per_concept
        
        # Create directories
        self.data_root.mkdir(exist_ok=True, parents=True)
        self.images_dir = self.data_root / "images"
        self.embeddings_dir = self.data_root / "embeddings"
        self.triplets_dir = self.data_root / "triplets"
        
        # Download and setup data
        if download:
            self.download_dataset()
        
        # Load dataset components
        self.concept_list = self.load_concept_list()
        self.image_paths = self.load_image_paths()
        self.hpe_embeddings = self.load_hpe_embeddings()
        
        if load_triplets:
            self.triplet_data = self.load_triplet_data()
            self.concept_triplets = self.prepare_concept_triplets()
        
        # Create splits
        self.create_splits()
        
        # Image transforms
        self.transform = self.create_transforms()
        
        logger.info(f"THINGS dataset loaded: {len(self)} samples in {split} split")
    
    def download_dataset(self):
        """Download THINGS dataset components"""
        logger.info("Downloading THINGS dataset...")
        
        # THINGS dataset URLs
        urls = {
            'images': 'https://files.osf.io/v1/resources/jum2f/providers/osfstorage/5e9c2a1613ba8300f1c31b28',
            'spose_embeddings': 'https://files.osf.io/v1/resources/jum2f/providers/osfstorage/5f73fbb7cb8d64001d8caac4', 
            'triplets': 'https://files.osf.io/v1/resources/jum2f/providers/osfstorage/5e9c2b1b24bb4400f88de1be',
            'concept_names': 'https://files.osf.io/v1/resources/jum2f/providers/osfstorage/5e9c39aa13ba8300f3c31c14'
        }
        
        # Download each component
        for component, url in urls.items():
            self._download_component(component, url)
    
    def _download_component(self, component: str, url: str):
        """Download a specific component of the dataset"""
        if component == 'images':
            target_dir = self.images_dir
            filename = 'images.zip'
        elif component == 'spose_embeddings':
            target_dir = self.embeddings_dir
            filename = 'spose_embeddings.npy'
        elif component == 'triplets':
            target_dir = self.triplets_dir
            filename = 'triplets.csv'
        elif component == 'concept_names':
            target_dir = self.data_root
            filename = 'concepts.txt'
        else:
            logger.warning(f"Unknown component: {component}")
            return
        
        target_dir.mkdir(exist_ok=True, parents=True)
        filepath = target_dir / filename
        
        if filepath.exists():
            logger.info(f"{component} already exists, skipping download")
            return
        
        logger.info(f"Downloading {component}...")
        
        try:
            # Download with progress bar
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as f, tqdm(
                desc=f"Downloading {component}",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
            
            # Extract if zip file
            if component == 'images' and filepath.suffix == '.zip':
                logger.info("Extracting images...")
                with zipfile.ZipFile(filepath, 'r') as zip_ref:
                    zip_ref.extractall(self.images_dir)
                filepath.unlink()  # Remove zip file
            
            logger.info(f"{component} downloaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to download {component}: {e}")
            if filepath.exists():
                filepath.unlink()
    
    def load_concept_list(self) -> List[str]:
        """Load list of THINGS concepts"""
        concepts_file = self.data_root / 'concepts.txt'
        
        if not concepts_file.exists():
            # Create dummy concept list if file doesn't exist
            logger.warning("Concepts file not found, creating dummy list")
            concepts = [f"concept_{i:04d}" for i in range(1854)]  # THINGS has 1854 concepts
        else:
            with open(concepts_file, 'r') as f:
                concepts = [line.strip() for line in f if line.strip()]
        
        return concepts
    
    def load_image_paths(self) -> Dict[str, str]:
        """Load mapping from concepts to image paths"""
        image_paths = {}
        
        # Look for images in the images directory
        if self.images_dir.exists():
            for concept in self.concept_list:
                # THINGS images are typically named as concept_name.jpg or concept_name.png
                possible_extensions = ['.jpg', '.jpeg', '.png', '.JPEG', '.JPG', '.PNG']
                
                for ext in possible_extensions:
                    img_path = self.images_dir / f"{concept}{ext}"
                    if img_path.exists():
                        image_paths[concept] = str(img_path)
                        break
                else:
                    # If no image found, create a dummy path
                    logger.debug(f"No image found for concept: {concept}")
        
        logger.info(f"Found {len(image_paths)} concept images")
        return image_paths
    
    def load_hpe_embeddings(self) -> Dict[str, np.ndarray]:
        """Load Human Perceptual Embeddings (SPoSE) for concepts"""
        embeddings_file = self.embeddings_dir / 'spose_embeddings.npy'
        
        if embeddings_file.exists():
            # Load pre-computed SPoSE embeddings
            embeddings_matrix = np.load(embeddings_file)
            
            # Map to concepts (assuming order matches concept list)
            embeddings = {}
            for i, concept in enumerate(self.concept_list):
                if i < embeddings_matrix.shape[0]:
                    embeddings[concept] = embeddings_matrix[i]
                else:
                    # Random embedding if not available
                    embeddings[concept] = np.random.randn(66)  # 66-dimensional HPE
            
            logger.info(f"Loaded HPE embeddings: {embeddings_matrix.shape}")
        else:
            # Generate random embeddings as placeholder
            logger.warning("HPE embeddings not found, generating random embeddings")
            embeddings = {}
            for concept in self.concept_list:
                embeddings[concept] = np.random.randn(66)
        
        return embeddings
    
    def load_triplet_data(self) -> pd.DataFrame:
        """Load triplet similarity judgments"""
        triplets_file = self.triplets_dir / 'triplets.csv'
        
        if triplets_file.exists():
            triplets = pd.read_csv(triplets_file)
            logger.info(f"Loaded {len(triplets)} triplet judgments")
        else:
            # Generate dummy triplet data
            logger.warning("Triplet data not found, generating dummy triplets")
            n_triplets = 10000
            triplets = pd.DataFrame({
                'anchor': np.random.choice(self.concept_list, n_triplets),
                'choice1': np.random.choice(self.concept_list, n_triplets),
                'choice2': np.random.choice(self.concept_list, n_triplets),
                'response': np.random.choice([1, 2], n_triplets)  # 1 or 2 for more similar choice
            })
        
        return triplets
    
    def prepare_concept_triplets(self) -> Dict[str, List[Tuple[str, str, str]]]:
        """Organize triplets by anchor concept"""
        concept_triplets = {concept: [] for concept in self.concept_list}
        
        for _, row in self.triplet_data.iterrows():
            anchor = row['anchor']
            
            if anchor not in concept_triplets:
                continue
            
            # Determine positive and negative based on response
            if row['response'] == 1:
                positive = row['choice1']
                negative = row['choice2']
            else:
                positive = row['choice2']
                negative = row['choice1']
            
            concept_triplets[anchor].append((anchor, positive, negative))
            
            # Limit triplets per concept
            if len(concept_triplets[anchor]) >= self.max_triplets_per_concept:
                continue
        
        # Filter out concepts with no triplets
        concept_triplets = {k: v for k, v in concept_triplets.items() if v}
        
        logger.info(f"Prepared triplets for {len(concept_triplets)} concepts")
        return concept_triplets
    
    def create_splits(self):
        """Create train/val/test splits"""
        # Get concepts that have both images and embeddings
        valid_concepts = [
            concept for concept in self.concept_list 
            if concept in self.image_paths and concept in self.hpe_embeddings
        ]
        
        # Create splits (80/10/10)
        np.random.seed(42)  # For reproducible splits
        np.random.shuffle(valid_concepts)
        
        n_total = len(valid_concepts)
        n_train = int(0.8 * n_total)
        n_val = int(0.1 * n_total)
        
        if self.split == 'train':
            self.split_concepts = valid_concepts[:n_train]
        elif self.split == 'val':
            self.split_concepts = valid_concepts[n_train:n_train + n_val]
        elif self.split == 'test':
            self.split_concepts = valid_concepts[n_train + n_val:]
        else:
            raise ValueError(f"Unknown split: {self.split}")
        
        logger.info(f"Split '{self.split}' contains {len(self.split_concepts)} concepts")
    
    def create_transforms(self) -> transforms.Compose:
        """Create image transforms for training"""
        if self.split == 'train':
            transform_list = [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [-1, 1] range
            ]
        else:
            transform_list = [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ]
        
        return transforms.Compose(transform_list)
    
    def __len__(self) -> int:
        return len(self.split_concepts)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get dataset item"""
        concept = self.split_concepts[idx]
        
        # Load and transform image
        try:
            image_path = self.image_paths[concept]
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            logger.warning(f"Failed to load image for {concept}: {e}")
            # Create dummy image
            image = torch.randn(3, self.image_size, self.image_size)
        
        # Get HPE embedding
        hpe_embedding = torch.tensor(self.hpe_embeddings[concept], dtype=torch.float32)
        
        # Prepare output
        item = {
            'images': image,
            'hpe_embeddings': hpe_embedding,
            'concept_name': concept,
            'concept_idx': self.concept_list.index(concept)
        }
        
        # Add triplet data if available
        if self.load_triplets and concept in self.concept_triplets:
            triplets = self.concept_triplets[concept]
            if triplets:
                # Randomly sample a triplet
                triplet_idx = np.random.randint(len(triplets))
                anchor, positive, negative = triplets[triplet_idx]
                
                # Get embeddings for triplet
                item['triplet_anchor'] = torch.tensor(self.hpe_embeddings[anchor], dtype=torch.float32)
                item['triplet_positive'] = torch.tensor(self.hpe_embeddings[positive], dtype=torch.float32)
                item['triplet_negative'] = torch.tensor(self.hpe_embeddings[negative], dtype=torch.float32)
        
        return item


def create_things_dataloaders(
    data_root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 256,
    download: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create THINGS dataset dataloaders for train/val/test
    
    Args:
        data_root: Root directory for dataset
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        image_size: Target image size
        download: Whether to download dataset
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = THINGSDataset(
        data_root=data_root,
        split='train',
        image_size=image_size,
        download=download,
        load_triplets=True
    )
    
    val_dataset = THINGSDataset(
        data_root=data_root,
        split='val',
        image_size=image_size,
        download=False,
        load_triplets=True
    )
    
    test_dataset = THINGSDataset(
        data_root=data_root,
        split='test',
        image_size=image_size,
        download=False,
        load_triplets=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader, test_loader


class THINGSDataModule:
    """
    Lightning-style data module for THINGS dataset
    """
    
    def __init__(
        self,
        data_root: str,
        batch_size: int = 32,
        num_workers: int = 4,
        image_size: int = 256,
        download: bool = True
    ):
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.download = download
        
    def setup(self):
        """Setup datasets"""
        self.train_loader, self.val_loader, self.test_loader = create_things_dataloaders(
            data_root=self.data_root,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            image_size=self.image_size,
            download=self.download
        )
    
    def train_dataloader(self):
        return self.train_loader
    
    def val_dataloader(self):
        return self.val_loader
    
    def test_dataloader(self):
        return self.test_loader


if __name__ == "__main__":
    # Test the dataset
    print("Testing THINGS Dataset...")
    
    # Create dataset
    dataset = THINGSDataset(
        data_root="./data/things",
        split='train',
        image_size=256,
        download=False,  # Set to True to download
        load_triplets=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test a sample
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample keys: {list(sample.keys())}")
        print(f"Image shape: {sample['images'].shape}")
        print(f"HPE embedding shape: {sample['hpe_embeddings'].shape}")
        print(f"Concept: {sample['concept_name']}")
        
        # Test dataloader
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        batch = next(iter(dataloader))
        print(f"Batch image shape: {batch['images'].shape}")
        print(f"Batch HPE shape: {batch['hpe_embeddings'].shape}")
