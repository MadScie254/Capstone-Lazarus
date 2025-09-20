"""
CAPSTONE-LAZARUS: Data Utilities
===============================
Comprehensive data loading, preprocessing, and augmentation utilities for plant disease detection.
"""

import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import img_to_array
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
try:
    import albumentations as A
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    print("⚠️ Albumentations not available. Using basic augmentations only.")
    A = None
    ALBUMENTATIONS_AVAILABLE = False
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter

class PlantDiseaseDataLoader:
    """Advanced data loader for plant disease images with comprehensive preprocessing."""
    
    def __init__(self, data_dir: str, img_size: Tuple[int, int] = (224, 224), batch_size: int = 32):
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.batch_size = batch_size
        self.class_names = []
        self.class_counts = {}
        
    def scan_dataset(self) -> Dict[str, Any]:
        """Scan dataset and return comprehensive statistics."""
        print("🔍 Scanning dataset...")
        
        stats = {
            'total_images': 0,
            'num_classes': 0,
            'class_distribution': {},
            'class_names': [],
            'imbalance_ratio': 0.0,
            'crop_types': {'corn': 0, 'potato': 0, 'tomato': 0, 'healthy': 0, 'diseased': 0}
        }
        
        # Scan all subdirectories
        for class_dir in self.data_dir.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.JPG')) + list(class_dir.glob('*.png'))
                count = len(image_files)
                
                if count > 0:
                    stats['class_distribution'][class_name] = count
                    stats['class_names'].append(class_name)
                    stats['total_images'] += count
                    
                    # Categorize by crop type
                    if 'corn' in class_name.lower() or 'maize' in class_name.lower():
                        stats['crop_types']['corn'] += count
                    elif 'potato' in class_name.lower():
                        stats['crop_types']['potato'] += count
                    elif 'tomato' in class_name.lower():
                        stats['crop_types']['tomato'] += count
                    
                    # Categorize by health status
                    if 'healthy' in class_name.lower():
                        stats['crop_types']['healthy'] += count
                    else:
                        stats['crop_types']['diseased'] += count
        
        stats['num_classes'] = len(stats['class_names'])
        
        # Calculate imbalance ratio
        if stats['class_distribution']:
            counts = list(stats['class_distribution'].values())
            stats['imbalance_ratio'] = max(counts) / min(counts)
        
        self.class_names = stats['class_names']
        self.class_counts = stats['class_distribution']
        
        print(f"✅ Dataset scan complete:")
        print(f"   📊 Total Images: {stats['total_images']:,}")
        print(f"   🏷️  Classes: {stats['num_classes']}")
        print(f"   ⚖️  Imbalance Ratio: {stats['imbalance_ratio']:.2f}")
        
        return stats
    
    def get_class_names(self) -> List[str]:
        """Get list of class names."""
        if not self.class_names:
            self.scan_dataset()
        return self.class_names
    
    def create_balanced_splits(self, test_size: float = 0.2, val_size: float = 0.2, 
                             random_state: int = 42) -> Tuple[Tuple[List, List], Tuple[List, List], Tuple[List, List]]:
        """Create stratified train/val/test splits maintaining class balance."""
        print("📊 Creating balanced dataset splits...")
        
        # Ensure class names are populated
        if not self.class_names:
            self.scan_dataset()
        
        all_paths = []
        all_labels = []
        
        # Collect all image paths and labels
        for idx, class_name in enumerate(self.class_names):
            class_dir = self.data_dir / class_name
            image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.JPG')) + list(class_dir.glob('*.png'))
            
            for img_path in image_files:
                all_paths.append(str(img_path))
                all_labels.append(idx)
        
        # Create stratified splits
        X_temp, X_test, y_temp, y_test = train_test_split(
            all_paths, all_labels, test_size=test_size, 
            stratify=all_labels, random_state=random_state
        )
        
        # Split remaining into train and validation
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted,
            stratify=y_temp, random_state=random_state
        )
        
        print(f"✅ Dataset splits created:")
        print(f"   🚂 Train: {len(X_train):,} images")
        print(f"   🔍 Validation: {len(X_val):,} images")  
        print(f"   🧪 Test: {len(X_test):,} images")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def compute_class_weights(self, y_train: List[int]) -> Dict[int, float]:
        """Compute class weights for handling imbalanced dataset."""
        class_weights = compute_class_weight(
            'balanced', 
            classes=np.unique(y_train), 
            y=y_train
        )
        return dict(zip(np.unique(y_train), class_weights))
    
    def get_augmentation_pipeline(self, is_training: bool = True):
        """Advanced augmentation pipeline using Albumentations or TensorFlow."""
        if not ALBUMENTATIONS_AVAILABLE:
            return None  # Will use TensorFlow augmentation in create_tf_dataset
            
        if is_training:
            return A.Compose([
                # Geometric transformations
                A.RandomRotate90(p=0.3),
                A.HorizontalFlip(p=0.3),  # Fixed from A.Flip
                A.Rotate(limit=15, p=0.3),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.3),
                
                # Color & lighting (simulate field conditions)
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.3),
                A.RandomGamma(gamma_limit=(80, 120), p=0.2),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(4, 4), p=0.2),
                
                # Noise & blur (simulate camera/environmental conditions)
                A.GaussNoise(var_limit=(10, 50), p=0.2),
                A.GaussianBlur(blur_limit=3, p=0.1),
                A.MotionBlur(blur_limit=3, p=0.1),
                
                # Occlusion simulation
                A.CoarseDropout(max_holes=8, max_height=32, max_width=32, 
                              min_holes=1, min_height=8, min_width=8, p=0.2),
                
                # Final resize and normalize
                A.Resize(self.img_size[0], self.img_size[1]),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))  # Fixed to tuples
            ])
        else:
            return A.Compose([
                A.Resize(self.img_size[0], self.img_size[1]),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))  # Fixed to tuples
            ])
    
    def create_tf_dataset(self, paths: List[str], labels: List[int], 
                         is_training: bool = True, shuffle: bool = True) -> tf.data.Dataset:
        """Create optimized TensorFlow dataset with augmentation."""
        
        def load_and_preprocess(path, label):
            # Load image
            image = tf.io.read_file(path)
            image = tf.image.decode_image(image, channels=3, expand_animations=False)
            image = tf.cast(image, tf.float32)
            
            # Resize
            image = tf.image.resize(image, self.img_size)
            
            # Normalize to [0, 1]
            image = image / 255.0
            
            # Additional augmentation for training (only if albumentations not available)
            if is_training and not ALBUMENTATIONS_AVAILABLE:
                # Random flip
                image = tf.image.random_flip_left_right(image)
                image = tf.image.random_flip_up_down(image)
                
                # Random brightness and contrast
                image = tf.image.random_brightness(image, max_delta=0.2)
                image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
                
                # Random hue and saturation
                image = tf.image.random_hue(image, max_delta=0.1)
                image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
            
            # Final normalization (ImageNet stats)
            mean = tf.constant([0.485, 0.456, 0.406])
            std = tf.constant([0.229, 0.224, 0.225])
            image = (image - mean) / std
            
            return image, label
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=min(len(paths), 10000))
        
        dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset

def analyze_image_quality(data_dir: str, sample_size: int = 100) -> Dict[str, Any]:
    """Analyze image quality metrics across dataset."""
    print(f"🔍 Analyzing image quality (sampling {sample_size} images)...")
    
    quality_metrics = {
        'resolutions': [],
        'file_sizes': [],
        'aspect_ratios': [],
        'brightness': [],
        'contrast': [],
        'sharpness': []
    }
    
    data_path = Path(data_dir)
    all_images = []
    
    # Collect all image paths
    for class_dir in data_path.iterdir():
        if class_dir.is_dir():
            images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.JPG')) + list(class_dir.glob('*.png'))
            all_images.extend(images)
    
    # Sample images for analysis
    if len(all_images) > sample_size:
        sample_images = random.sample(all_images, sample_size)
    else:
        sample_images = all_images
    
    for img_path in sample_images:
        try:
            # Load image
            img = cv2.imread(str(img_path))
            if img is None:
                continue
                
            h, w, c = img.shape
            
            # Resolution and aspect ratio
            quality_metrics['resolutions'].append(f"{w}x{h}")
            quality_metrics['aspect_ratios'].append(w/h)
            
            # File size
            file_size = os.path.getsize(img_path) / 1024  # KB
            quality_metrics['file_sizes'].append(file_size)
            
            # Convert to grayscale for quality metrics
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Brightness (mean pixel value)
            brightness = float(np.mean(gray))
            quality_metrics['brightness'].append(brightness)
            
            # Contrast (standard deviation)
            contrast = float(np.std(gray))
            quality_metrics['contrast'].append(contrast)
            
            # Sharpness (Laplacian variance)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = laplacian.var()
            quality_metrics['sharpness'].append(sharpness)
            
        except Exception as e:
            print(f"⚠️  Error processing {img_path}: {e}")
            continue
    
    # Calculate statistics
    stats = {}
    for metric, values in quality_metrics.items():
        if values and metric != 'resolutions':
            stats[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values)
            }
    
    # Resolution distribution
    resolution_counts = Counter(quality_metrics['resolutions'])
    stats['top_resolutions'] = dict(resolution_counts.most_common(5))
    
    print("✅ Image quality analysis complete!")
    
    return stats

    def get_dataset_stats(self, compute_image_shape: bool = False, sample_max: int = 500) -> Dict[str, Any]:
        """Get comprehensive dataset statistics with defensive programming.
        
        Args:
            compute_image_shape: Whether to compute mean image shape (can be slow)
            sample_max: Maximum number of images to sample for shape computation
            
        Returns:
            Dict containing: total_images, valid_images, corrupted_images, num_classes,
            class_distribution, class_names, imbalance_ratio, mean_image_shape, dataframe
            
        Raises:
            FileNotFoundError: If data directory doesn't exist
        """
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        print("🔍 Computing comprehensive dataset statistics...")
        
        stats = {
            'total_images': 0,
            'valid_images': 0,
            'corrupted_images': 0,
            'num_classes': 0,
            'class_distribution': {},
            'class_names': [],
            'imbalance_ratio': 0.0,
            'mean_image_shape': None,
            'dataframe': None
        }
        
        image_data = []
        corrupted_files = []
        
        # Scan all subdirectories
        for class_dir in self.data_dir.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.JPG')) + list(class_dir.glob('*.png'))
                valid_count = 0
                
                for img_path in image_files:
                    stats['total_images'] += 1
                    
                    # Check if image is valid
                    try:
                        with Image.open(img_path) as img:
                            img.verify()  # Verify it's a valid image
                        valid_count += 1
                        stats['valid_images'] += 1
                        
                        # Add to dataframe data
                        image_data.append({
                            'image_path': str(img_path),
                            'class_name': class_name,
                            'filename': img_path.name,
                            'file_size_mb': img_path.stat().st_size / (1024 * 1024)
                        })
                        
                    except (OSError, IOError) as e:
                        print(f"⚠️ Corrupted image: {img_path} - {e}")
                        corrupted_files.append(str(img_path))
                        stats['corrupted_images'] += 1
                
                if valid_count > 0:
                    stats['class_distribution'][class_name] = valid_count
                    stats['class_names'].append(class_name)
        
        stats['num_classes'] = len(stats['class_names'])
        
        # Calculate imbalance ratio
        if stats['class_distribution']:
            counts = list(stats['class_distribution'].values())
            stats['imbalance_ratio'] = max(counts) / min(counts)
        
        # Create DataFrame
        stats['dataframe'] = pd.DataFrame(image_data)
        
        # Compute mean image shape if requested
        if compute_image_shape and image_data:
            print(f"📐 Computing mean image shape from {min(len(image_data), sample_max)} samples...")
            sample_data = random.sample(image_data, min(len(image_data), sample_max))
            shapes = []
            
            for item in sample_data:
                try:
                    with Image.open(item['image_path']) as img:
                        shapes.append(img.size[::-1] + (len(img.getbands()),))  # (H, W, C)
                except Exception as e:
                    print(f"⚠️ Could not read shape for {item['image_path']}: {e}")
                    continue
            
            if shapes:
                mean_shape = tuple(int(np.mean([s[i] for s in shapes])) for i in range(len(shapes[0])))
                stats['mean_image_shape'] = mean_shape
        
        # Update instance variables
        self.class_names = stats['class_names']
        self.class_counts = stats['class_distribution']
        
        print(f"✅ Dataset statistics complete:")
        print(f"   📊 Total Images: {stats['total_images']:,}")
        print(f"   ✅ Valid Images: {stats['valid_images']:,}")
        print(f"   ❌ Corrupted Images: {stats['corrupted_images']}")
        print(f"   🏷️  Classes: {stats['num_classes']}")
        print(f"   ⚖️  Imbalance Ratio: {stats['imbalance_ratio']:.2f}")
        if stats['mean_image_shape']:
            print(f"   📐 Mean Image Shape: {stats['mean_image_shape']}")
        
        return stats
    
    def analyze_class_distribution(self) -> Dict[str, int]:
        """Analyze class distribution and return counts."""
        if not self.class_counts:
            self.scan_dataset()
        return self.class_counts
    
    def get_all_image_paths_and_labels(self):
        """Get all image paths and their corresponding labels."""
        all_paths = []
        all_labels = []
        
        if not self.class_names:
            self.scan_dataset()
        
        # Collect all image paths and labels
        for idx, class_name in enumerate(self.class_names):
            class_dir = self.data_dir / class_name
            image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.JPG')) + list(class_dir.glob('*.png'))
            
            for img_path in image_files:
                all_paths.append(str(img_path))
                all_labels.append(idx)
        
        return all_paths, all_labels
    
    def visualize_class_distribution(self, save_path: Optional[str] = None):
        """Create interactive visualizations of class distribution."""
        if not self.class_counts:
            self.scan_dataset()
        
        # Create DataFrame for visualization
        df = pd.DataFrame([
            {'Class': class_name, 'Count': count, 'Plant_Type': self._get_plant_type(class_name)}
            for class_name, count in self.class_counts.items()
        ])
        
        # Interactive bar chart
        fig = px.bar(
            df,
            x='Class',
            y='Count',
            color='Plant_Type',
            title=f'Plant Disease Dataset Distribution - {sum(self.class_counts.values()):,} Total Images',
            labels={'Count': 'Number of Images'},
            height=600
        )
        fig.update_layout(xaxis_tickangle=-45)
        
        if save_path:
            fig.write_html(save_path)
        else:
            fig.show()
        
        return fig
    
    def _get_plant_type(self, class_name: str) -> str:
        """Extract plant type from class name."""
        class_lower = class_name.lower()
        if 'corn' in class_lower or 'maize' in class_lower:
            return 'Corn'
        elif 'potato' in class_lower:
            return 'Potato'
        elif 'tomato' in class_lower:
            return 'Tomato'
        else:
            return 'Other'

if __name__ == "__main__":
    # Example usage
    loader = PlantDiseaseDataLoader("data", img_size=(224, 224), batch_size=32)
    stats = loader.scan_dataset()
    
    # Create splits
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = loader.create_balanced_splits()
    
    # Compute class weights
    class_weights = loader.compute_class_weights(y_train)
    
    print(f"Class weights: {class_weights}")