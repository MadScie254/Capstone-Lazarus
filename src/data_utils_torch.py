"""
CAPSTONE-LAZARUS: PyTorch Data Utils with Albumentations
======================================================
Fast, efficient data loading optimized for HP ZBook G5 and Colab scaling.
"""

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from pathlib import Path
import numpy as np
from PIL import Image
import logging
from typing import Tuple, Optional, Dict, Any, List
import random

logger = logging.getLogger(__name__)

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    A = None  # Define A as None when not available
    ToTensorV2 = None
    logger.warning("Albumentations not available, using torchvision transforms")


class PlantDiseaseDataset(Dataset):
    """
    Plant Disease Dataset with Albumentations support.
    Optimized for fast loading and memory efficiency.
    """
    
    def __init__(
        self,
        root_dir: str,
        transform: Optional[Any] = None,
        use_albumentations: bool = True
    ):
        self.root_dir = Path(root_dir)
        self.use_albumentations = use_albumentations and ALBUMENTATIONS_AVAILABLE
        self.transform = transform
        
        # Load dataset using ImageFolder for class mapping
        self.dataset = ImageFolder(str(root_dir))
        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx
        self.samples = self.dataset.samples
        
        logger.info(f"Dataset loaded: {len(self.samples)} images, {len(self.classes)} classes")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            if self.use_albumentations:
                # Convert PIL to numpy for Albumentations
                image = np.array(image)
                transformed = self.transform(image=image)
                image = transformed['image']
            else:
                # Standard torchvision transforms
                image = self.transform(image)
        else:
            # Default: convert to tensor
            image = transforms.ToTensor()(image)
            
        return image, label


def get_albumentations_transforms(
    image_size: int = 224,
    split: str = "train",
    strength: str = "medium"
) -> Any:
    """
    Get Albumentations transforms for different training phases.
    
    Args:
        image_size: Target image size
        split: 'train', 'val', or 'test'
        strength: 'light', 'medium', 'heavy' augmentation strength
    
    Returns:
        Albumentations Compose object
    """
    
    if not ALBUMENTATIONS_AVAILABLE:
        raise ImportError("Albumentations required. Install with: pip install albumentations")
    
    # Base transforms
    base_transforms = [
        A.Resize(image_size, image_size, always_apply=True),
    ]
    
    if split == "train":
        # Training augmentations based on strength
        if strength == "light":
            aug_transforms = [
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=15, p=0.3),
                A.RandomBrightnessContrast(
                    brightness_limit=0.1,
                    contrast_limit=0.1,
                    p=0.3
                ),
            ]
        elif strength == "medium":
            aug_transforms = [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.Rotate(limit=25, p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5
                ),
                A.HueSaturationValue(
                    hue_shift_limit=10,
                    sat_shift_limit=20,
                    val_shift_limit=10,
                    p=0.3
                ),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,
                    rotate_limit=15,
                    p=0.3
                ),
            ]
        elif strength == "heavy":
            aug_transforms = [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.Rotate(limit=35, p=0.6),
                A.RandomBrightnessContrast(
                    brightness_limit=0.3,
                    contrast_limit=0.3,
                    p=0.6
                ),
                A.HueSaturationValue(
                    hue_shift_limit=20,
                    sat_shift_limit=30,
                    val_shift_limit=20,
                    p=0.5
                ),
                A.ShiftScaleRotate(
                    shift_limit=0.15,
                    scale_limit=0.15,
                    rotate_limit=25,
                    p=0.5
                ),
                A.OneOf([
                    A.ElasticTransform(p=0.3),
                    A.GridDistortion(p=0.3),
                    A.OpticalDistortion(p=0.3),
                ], p=0.3),
                A.OneOf([
                    A.GaussNoise(p=0.3),
                    A.GaussianBlur(blur_limit=3, p=0.3),
                    A.MotionBlur(blur_limit=3, p=0.3),
                ], p=0.2),
            ]
        else:
            raise ValueError(f"Unknown strength: {strength}")
            
        transforms_list = base_transforms + aug_transforms
    else:
        # Validation/test transforms (no augmentation)
        transforms_list = base_transforms
    
    # Add normalization and tensor conversion
    transforms_list.extend([
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            always_apply=True
        ),
        ToTensorV2(always_apply=True)
    ])
    
    return A.Compose(transforms_list)


def get_torchvision_transforms(
    image_size: int = 224,
    split: str = "train"
) -> transforms.Compose:
    """
    Get torchvision transforms as fallback.
    """
    
    if split == "train":
        transform_list = [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(25),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]
    else:
        transform_list = [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]
    
    return transforms.Compose(transform_list)


def create_weighted_sampler(dataset) -> WeightedRandomSampler:
    """
    Create weighted sampler for class balancing.
    Handles both full datasets and Subset objects.
    """
    from torch.utils.data import Subset
    
    # Get class counts - handle Subset vs full dataset
    if isinstance(dataset, Subset):
        # For Subset, access the underlying dataset
        labels = [dataset.dataset.samples[idx][1] for idx in dataset.indices]
    elif hasattr(dataset, 'samples'):
        # For ImageFolder or PlantDiseaseDataset
        labels = [sample[1] for sample in dataset.samples]
    else:
        # Fallback: iterate through dataset
        labels = [dataset[i][1] for i in range(len(dataset))]
    
    class_counts = np.bincount(labels)
    
    # Calculate weights (inverse frequency)
    num_samples = len(labels)
    class_weights = num_samples / (len(class_counts) * class_counts)
    
    # Assign weight to each sample
    sample_weights = [class_weights[label] for label in labels]
    
    logger.info(f"Class distribution: {dict(enumerate(class_counts))}")
    logger.info(f"Class weights: {dict(enumerate(class_weights))}")
    
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )


def make_dataloaders(
    data_dir: str,
    config: Dict[str, Any],
    train_split: float = 0.8,
    val_split: float = 0.2
) -> Tuple[DataLoader, DataLoader]:
    """
    NUCLEAR OPTIMIZATION: Create super-fast dataloaders.
    
    Args:
        data_dir: Root directory containing class subdirectories
        config: Configuration dictionary
        train_split: Fraction for training
        val_split: Fraction for validation
        
    Returns:
        (train_loader, val_loader)
    """
    
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # NUCLEAR: Force fast settings
    image_size = min(config.get('image_size', 128), 128)  # Cap at 128
    batch_size = max(config.get('batch_size', 32), 16)    # Min 16
    num_workers = min(config.get('num_workers', 2), 2)    # Max 2
    
    print(f"🚨 NUCLEAR DATALOADERS: img={image_size}, batch={batch_size}, workers={num_workers}")
    
    # NUCLEAR: Simple transforms only
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(), 
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    full_dataset = ImageFolder(root=str(data_path), transform=train_transform)
    print(f"✓ Dataset: {len(full_dataset)} samples, {len(full_dataset.classes)} classes")
    
    # SMART COMPROMISE: Use more data unless fast_test
    if config.get('fast_test', False):
        subset_size = min(500, len(full_dataset) // 10)  # 10% for fast test
        indices = torch.randperm(len(full_dataset))[:subset_size].tolist()
        full_dataset = torch.utils.data.Subset(full_dataset, indices)
        print(f"🚨 FAST TEST: Using {len(full_dataset)} samples")
    else:
        # SMART: Use 20% of data for better accuracy
        subset_size = min(2000, len(full_dataset) // 5)  # 20% max 2000 samples
        if subset_size < len(full_dataset):
            indices = torch.randperm(len(full_dataset))[:subset_size].tolist()
            full_dataset = torch.utils.data.Subset(full_dataset, indices)
            print(f"🎯 SMART MODE: Using {len(full_dataset)} samples (20%)")
        else:
            print(f"✓ Using full dataset: {len(full_dataset)} samples")
    
    # Split
    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Apply val transforms
    if hasattr(val_dataset, 'dataset'):
        val_dataset.dataset.transform = val_transform
    
    # NUCLEAR: Simple dataloaders
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
        pin_memory=True
    )
    
    print(f"✓ Created: {len(train_loader)} train batches, {len(val_loader)} val batches")
    return train_loader, val_loader


def get_albumentations_transforms(image_size: int, split: str = 'train', strength: str = 'medium'):
    """Get albumentations transforms with specified strength."""
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    
    # Base transforms for all modes
    base_transforms = [
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ]
    
    if split == 'train':
        # Training augmentations based on strength
        if strength == 'light':
            train_transforms = [
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2)
            ] + base_transforms
        elif strength == 'medium':
            train_transforms = [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.RandomBrightnessContrast(p=0.3),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.3)
            ] + base_transforms
        else:  # heavy
            train_transforms = [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomBrightnessContrast(p=0.4),
                A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.4),
                A.HueSaturationValue(p=0.3)
            ] + base_transforms
        
        return A.Compose(train_transforms)
    else:
        # Validation/test transforms - no augmentation
        return A.Compose(base_transforms)


def get_torchvision_transforms(image_size: int, split: str = 'train'):
    """Get torchvision transforms as fallback."""
    from torchvision import transforms
    
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def make_existing_split_dataloaders(config: Dict[str, Any], data_root: Path) -> Tuple[DataLoader, DataLoader]:
    """Create dataloaders from existing train/val split directories."""
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    
    # Check if albumentations is available
    try:
        import albumentations
        use_albu = True
        train_transform = get_albumentations_transforms(
            image_size=config['image_size'],
            split='train',
            strength=config.get('augmentation_strength', 'medium')
        )
        val_transform = get_albumentations_transforms(
            image_size=config['image_size'],
            split='val'
        )
    except ImportError:
        use_albu = False
        train_transform = get_torchvision_transforms(
            image_size=config['image_size'],
            split='train'
        )
        val_transform = get_torchvision_transforms(
            image_size=config['image_size'],
            split='val'
        )
    
    # Create datasets
    train_dataset = PlantDiseaseDataset(
        train_dir, 
        transform=train_transform,
        use_albumentations=use_albu
    )
    val_dataset = PlantDiseaseDataset(
        val_dir,
        transform=val_transform,
        use_albumentations=use_albu
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )
    
    print(f"✓ Created from existing split: {len(train_loader)} train batches, {len(val_loader)} val batches")
    return train_loader, val_loader
        
        train_size = int(train_split * dataset_size)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        # Create subsets
        train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
        val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
        
        # Apply transforms to subsets
        train_dataset.dataset.transform = train_transform
        val_dataset.dataset.transform = val_transform
    
    # Create weighted sampler for training
    if config.get('use_class_balancing', True):
        train_sampler = create_weighted_sampler(train_dataset)
        shuffle = False  # Don't shuffle when using sampler
    else:
        train_sampler = None
        shuffle = True
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        sampler=train_sampler,
        shuffle=shuffle,
        num_workers=config.get('num_workers', 4),
        pin_memory=config.get('pin_memory', True),
        persistent_workers=True if config.get('num_workers', 4) > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=config.get('pin_memory', True),
        persistent_workers=True if config.get('num_workers', 4) > 0 else False
    )
    
    logger.info(f"Train loader: {len(train_loader)} batches ({len(train_dataset)} samples)")
    logger.info(f"Val loader: {len(val_loader)} batches ({len(val_dataset)} samples)")
    
    return train_loader, val_loader


def create_subset_loader(
    data_dir: str,
    config: Dict[str, Any],
    subset_size: int = 1000,
    split: str = "train"
) -> DataLoader:
    """
    Create a DataLoader with a subset of data for quick testing.
    
    Args:
        data_dir: Data directory path
        config: Configuration dictionary
        subset_size: Number of samples to include
        split: 'train' or 'val'
        
    Returns:
        DataLoader with subset of data
    """
    
    # Get appropriate transforms
    if ALBUMENTATIONS_AVAILABLE and config.get('use_augmentations', True):
        transform = get_albumentations_transforms(
            image_size=config['image_size'],
            split=split,
            strength=config.get('augmentation_strength', 'light')  # Light for quick testing
        )
        use_albu = True
    else:
        transform = get_torchvision_transforms(
            image_size=config['image_size'],
            split=split
        )
        use_albu = False
    
    # Create dataset
    dataset = PlantDiseaseDataset(
        data_dir,
        transform=transform,
        use_albumentations=use_albu
    )
    
    # Create random subset
    subset_size = min(subset_size, len(dataset))
    indices = random.sample(range(len(dataset)), subset_size)
    subset_dataset = torch.utils.data.Subset(dataset, indices)
    
    # Create loader
    loader = DataLoader(
        subset_dataset,
        batch_size=config['batch_size'],
        shuffle=(split == 'train'),
        num_workers=config.get('num_workers', 2),  # Fewer workers for testing
        pin_memory=False  # Disable for quick testing
    )
    
    logger.info(f"Subset loader created: {len(loader)} batches ({subset_size} samples)")
    
    return loader


# Data analysis utilities
def analyze_dataset_distribution(data_dir: str) -> Dict[str, Any]:
    """Analyze class distribution in dataset."""
    
    dataset = ImageFolder(data_dir)
    
    # Count samples per class
    class_counts = {}
    for _, class_idx in dataset.samples:
        class_name = dataset.classes[class_idx]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    total_samples = len(dataset.samples)
    
    analysis = {
        'total_samples': total_samples,
        'num_classes': len(dataset.classes),
        'class_names': dataset.classes,
        'class_counts': class_counts,
        'class_percentages': {
            name: (count / total_samples) * 100 
            for name, count in class_counts.items()
        }
    }
    
    return analysis


if __name__ == "__main__":
    # Test data loading
    print("Testing data loading...")
    
    # Mock config for testing
    config = {
        'image_size': 224,
        'batch_size': 16,
        'num_workers': 2,
        'pin_memory': True,
        'use_augmentations': True,
        'augmentation_strength': 'medium',
        'seed': 42
    }
    
    # This would need actual data directory to run
    # train_loader, val_loader = make_dataloaders('./data', config)
    # print(f"Created loaders with {len(train_loader)} train batches, {len(val_loader)} val batches")
    
    print("Data utilities ready!")