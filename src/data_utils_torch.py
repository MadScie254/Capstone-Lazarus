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
    import random
    import torch.utils.data
    
    # Get appropriate transforms
    try:
        import albumentations
        transform = get_albumentations_transforms(
            image_size=config['image_size'],
            split=split,
            strength=config.get('augmentation_strength', 'light')  # Light for quick testing
        )
        use_albu = True
    except ImportError:
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
    
    # Create balanced subset
    # Estimate samples per class
    n_classes = len(dataset.classes)
    samples_per_class = max(5, subset_size // n_classes)
    
    subset_dataset = create_balanced_subset(dataset.dataset, samples_per_class=samples_per_class, total_max=subset_size)
    
    # Create loader
    loader = DataLoader(
        subset_dataset,
        batch_size=config['batch_size'],
        shuffle=(split == 'train'),
        num_workers=config.get('num_workers', 2),  # Fewer workers for testing
        pin_memory=False  # Disable for quick testing
    )
    
    print(f"✓ Balanced subset loader created: {len(loader)} batches ({subset_size} samples)")
    
    return loader