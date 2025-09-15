"""
Data ETL Pipeline for CAPSTONE-LAZARUS Plant Disease Detection
==============================================================
Advanced data processing pipeline specifically designed for agricultural 
image analysis and plant disease detection applications.
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import random
from collections import Counter
import json

from src.config import Config

logger = logging.getLogger(__name__)

class PlantDiseaseDataPipeline:
    """
    Advanced ETL pipeline specifically designed for plant disease detection
    
    Features:
    - Multi-crop disease classification support
    - Class imbalance handling with strategic sampling
    - Advanced agricultural image augmentation
    - Disease severity estimation
    - Crop-specific preprocessing
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.data_config = config.data
        self.label_encoder = LabelEncoder()
        self.class_weights = None
        self.disease_taxonomy = {}
        
        # Plant disease specific configurations
        self.crop_types = ['Corn', 'Potato', 'Tomato']
        self.healthy_classes = []
        self.disease_classes = []
        
    def analyze_dataset_structure(self, data_path: str) -> Dict[str, Any]:
        """Comprehensive analysis of plant disease dataset structure"""
        
        data_path_obj = Path(data_path)
        logger.info(f"Analyzing plant disease dataset: {data_path_obj}")
        
        if not data_path_obj.exists():
            raise FileNotFoundError(f"Data directory not found: {data_path_obj}")
        
        # Get all class directories
        class_dirs = [d for d in data_path_obj.iterdir() if d.is_dir()]
        class_info = []
        total_images = 0
        
        for class_dir in class_dirs:
            # Count images in each class
            image_files = self._get_image_files(class_dir)
            num_images = len(image_files)
            total_images += num_images
            
            # Parse plant disease class information
            class_name = class_dir.name
            crop, condition = self._parse_class_name(class_name)
            
            # Sample image for quality analysis
            sample_width, sample_height = 0, 0
            if image_files:
                try:
                    sample_img = cv2.imread(str(image_files[0]))
                    if sample_img is not None:
                        sample_height, sample_width = sample_img.shape[:2]
                except Exception as e:
                    logger.warning(f"Could not read sample image from {class_dir}: {e}")
            
            class_info.append({
                'class_name': class_name,
                'crop': crop,
                'condition': condition,
                'num_images': num_images,
                'sample_width': sample_width,
                'sample_height': sample_height,
                'directory': class_dir,
                'is_healthy': 'healthy' in condition.lower(),
                'severity': self._estimate_disease_severity(condition)
            })
        
        # Create comprehensive analysis
        analysis = {
            'total_classes': len(class_dirs),
            'total_images': total_images,
            'class_distribution': class_info,
            'crops': list(set([info['crop'] for info in class_info])),
            'imbalance_ratio': max([info['num_images'] for info in class_info]) / 
                              max(1, min([info['num_images'] for info in class_info])),
            'healthy_classes': [info for info in class_info if info['is_healthy']],
            'disease_classes': [info for info in class_info if not info['is_healthy']],
        }
        
        logger.info(f"Dataset analysis complete:")
        logger.info(f"  - Total classes: {analysis['total_classes']}")
        logger.info(f"  - Total images: {analysis['total_images']:,}")
        logger.info(f"  - Crops: {len(analysis['crops'])} ({', '.join(analysis['crops'])})")
        logger.info(f"  - Imbalance ratio: {analysis['imbalance_ratio']:.1f}:1")
        
        return analysis
    
    def _get_image_files(self, directory: Path) -> List[Path]:
        """Get all image files from directory"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.JPG', '.JPEG', '.PNG'}
        return [f for f in directory.iterdir() 
                if f.is_file() and f.suffix in image_extensions]
    
    def _parse_class_name(self, class_name: str) -> Tuple[str, str]:
        """Parse plant disease class name into crop and condition"""
        if '___' in class_name:
            crop_part, condition = class_name.split('___', 1)
            # Clean up crop name
            crop = crop_part.replace('(', '').replace(')', '').replace('_', ' ').strip()
            # Capitalize properly
            crop = ' '.join([word.capitalize() for word in crop.split()])
        else:
            crop = 'Unknown'
            condition = class_name
        
        return crop, condition
    
    def _estimate_disease_severity(self, condition: str) -> int:
        """Estimate disease severity from condition name (0=healthy, 1=mild, 2=moderate, 3=severe)"""
        condition_lower = condition.lower()
        
        if 'healthy' in condition_lower:
            return 0
        elif any(term in condition_lower for term in ['early', 'minor', 'light']):
            return 1
        elif any(term in condition_lower for term in ['late', 'severe', 'advanced']):
            return 3
        else:
            return 2  # moderate
    
    def prepare_plant_disease_datasets(self, data_path: str) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """
        Prepare plant disease datasets with agricultural-specific considerations
        
        Returns:
            Tuple of (train_ds, val_ds, test_ds)
        """
        
        logger.info(f"Preparing plant disease datasets from {data_path}")
        
        # Analyze dataset first
        analysis = self.analyze_dataset_structure(data_path)
        
        # Update config with detected information
        self.config.model.num_classes = analysis['total_classes']
        
        # Collect all image paths and labels
        image_paths, labels, metadata = self._collect_image_data(Path(data_path), analysis)
        
        # Encode labels
        labels_encoded_array = np.array(self.label_encoder.fit_transform(labels))
        
        # Calculate class weights for imbalanced dataset
        self.class_weights = self._calculate_agricultural_class_weights(
            labels_encoded_array, analysis
        )
        
        # Strategic dataset splitting for agricultural applications
        X_train, X_val, X_test, y_train, y_val, y_test = self._strategic_dataset_split(
            image_paths, labels_encoded_array, metadata
        )
        
        logger.info(f"Dataset split complete:")
        logger.info(f"  - Training: {len(X_train):,} images")
        logger.info(f"  - Validation: {len(X_val):,} images")
        logger.info(f"  - Test: {len(X_test):,} images")
        
        # Create TensorFlow datasets
        train_ds = self._create_agricultural_tf_dataset(np.array(X_train), y_train, training=True)
        val_ds = self._create_agricultural_tf_dataset(np.array(X_val), y_val, training=False)
        test_ds = self._create_agricultural_tf_dataset(np.array(X_test), y_test, training=False)
        
        # Store dataset information
        dataset_info = {
            'class_names': list(self.label_encoder.classes_),
            'class_weights': self.class_weights,
            'dataset_analysis': analysis,
            'splits': {
                'train_size': len(X_train),
                'val_size': len(X_val),
                'test_size': len(X_test)
            }
        }
        
        # Save dataset information
        self._save_dataset_info(dataset_info)
        
        logger.info("Plant disease datasets prepared successfully")
        return train_ds, val_ds, test_ds
    
    def _collect_image_data(self, data_path: Path, analysis: Dict) -> Tuple[List[str], List[str], List[Dict]]:
        """Collect all image paths, labels, and metadata"""
        
        image_paths = []
        labels = []
        metadata = []
        
        for class_info in analysis['class_distribution']:
            class_dir = class_info['directory']
            class_name = class_info['class_name']
            
            image_files = self._get_image_files(class_dir)
            
            for img_path in image_files:
                image_paths.append(str(img_path))
                labels.append(class_name)
                metadata.append({
                    'crop': class_info['crop'],
                    'condition': class_info['condition'],
                    'is_healthy': class_info['is_healthy'],
                    'severity': class_info['severity'],
                    'class_size': class_info['num_images']
                })
        
        return image_paths, labels, metadata
    
    def _calculate_agricultural_class_weights(self, labels_encoded: np.ndarray, analysis: Dict) -> Dict[int, float]:
        """Calculate class weights with agricultural priorities"""
        
        # Standard balanced weights
        unique_labels = np.unique(labels_encoded)
        balanced_weights = compute_class_weight('balanced', classes=unique_labels, y=labels_encoded)
        
        # Create class weight dictionary
        class_weights = dict(zip(unique_labels, balanced_weights))
        
        # Agricultural adjustments: boost critical disease classes
        for i, class_info in enumerate(analysis['class_distribution']):
            if not class_info['is_healthy'] and class_info['severity'] >= 2:
                # Boost weights for moderate to severe diseases
                class_weights[i] *= 1.2
        
        logger.info(f"Class weights calculated (range: {min(class_weights.values()):.3f} - {max(class_weights.values()):.3f})")
        
        return class_weights
    
    def _strategic_dataset_split(self, image_paths: List[str], labels_encoded: np.ndarray, 
                               metadata: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                                            np.ndarray, np.ndarray, np.ndarray]:
        """Strategic dataset splitting for agricultural applications"""
        
        # Convert to arrays for easier manipulation
        image_paths_array = np.array(image_paths)
        metadata_df = pd.DataFrame(metadata)
        
        # Stratified split by crop and disease status to ensure representation
        stratify_labels = [f"{meta['crop']}_{meta['is_healthy']}" for meta in metadata]
        
        # First split: separate training from validation+test
        X_train, X_temp, y_train, y_temp, strat_train, strat_temp = train_test_split(
            image_paths_array, labels_encoded, stratify_labels,
            test_size=self.data_config.validation_split + self.data_config.test_split,
            random_state=self.config.random_seed,
            stratify=stratify_labels
        )
        
        # Second split: separate validation from test
        test_size_ratio = self.data_config.test_split / (
            self.data_config.validation_split + self.data_config.test_split
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=test_size_ratio,
            random_state=self.config.random_seed,
            stratify=strat_temp
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def _create_agricultural_tf_dataset(self, image_paths: np.ndarray, labels: np.ndarray, 
                                      training: bool = False) -> tf.data.Dataset:
        """Create TensorFlow dataset with agricultural-specific preprocessing"""
        
        def load_and_preprocess_image(path, label):
            """Load and preprocess plant disease images"""
            
            # Load image
            image = tf.io.read_file(path)
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.image.resize(image, self.data_config.image_size)
            image = tf.cast(image, tf.float32) / 255.0
            
            if training and self.data_config.augmentation:
                # Agricultural-specific augmentations
                image = self._apply_agricultural_augmentation(image)
            
            # Convert label to categorical
            label = tf.one_hot(label, depth=self.config.model.num_classes)
            
            return image, label
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
        dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        
        # Performance optimizations
        if training:
            dataset = dataset.shuffle(self.data_config.shuffle_buffer)
            dataset = dataset.repeat()
        
        dataset = dataset.batch(self.data_config.batch_size)
        
        if self.data_config.cache:
            dataset = dataset.cache()
        
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def _apply_agricultural_augmentation(self, image):
        """Apply agricultural-specific image augmentation"""
        
        # Standard augmentations
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        
        # Color augmentations (simulate different lighting conditions)
        image = tf.image.random_brightness(image, max_delta=0.15)
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
        image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
        image = tf.image.random_hue(image, max_delta=0.08)
        
        # Geometric augmentations
        image = tf.image.rot90(image, tf.random.uniform(shape=[], maxval=4, dtype=tf.int32))
        
        # Random crop and zoom (simulate different distances/angles)
        zoom_factor = tf.random.uniform([], 1.0, 1.15)
        new_size = tf.cast(tf.cast(self.data_config.image_size, tf.float32) * zoom_factor, tf.int32)
        image = tf.image.resize(image, new_size)
        image = tf.image.random_crop(image, size=[*self.data_config.image_size, 3])
        
        # Simulate field conditions
        if tf.random.uniform([]) < 0.1:
            # Add slight blur (simulate motion or focus issues)
            kernel_size = tf.random.uniform([], 1, 3, dtype=tf.int32)
            try:
                # Some TF versions may not have gaussian_blur
                image = tf.image.gaussian_blur(image, [kernel_size, kernel_size], [0.5, 0.5])
            except Exception:
                # Fallback: average pooling as mild blur
                k = tf.maximum(kernel_size, 1)
                k = tf.cast(k, tf.int32)
                image = tf.nn.avg_pool2d(tf.expand_dims(image, 0), ksize=k, strides=1, padding='SAME')
                image = tf.squeeze(image, 0)
        
        return image
    
    def _save_dataset_info(self, dataset_info: Dict):
        """Save dataset information for later use"""
        
        # Create data directory if it doesn't exist
        data_dir = Path("data_info")
        data_dir.mkdir(exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        json_info = {}
        for key, value in dataset_info.items():
            if isinstance(value, np.ndarray):
                json_info[key] = value.tolist()
            elif isinstance(value, dict):
                json_info[key] = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                                for k, v in value.items()}
            else:
                json_info[key] = value
        
        with open(data_dir / "dataset_info.json", "w") as f:
            json.dump(json_info, f, indent=2, default=str)
        
        logger.info(f"Dataset information saved to {data_dir}/dataset_info.json")
    
    def get_class_names(self) -> List[str]:
        """Get class names in order"""
        return list(self.label_encoder.classes_) if hasattr(self.label_encoder, 'classes_') else []
    
    def get_class_weights(self) -> Dict[int, float]:
        """Get calculated class weights"""
        return self.class_weights or {}


# Backward-compatible wrapper expected by the app and CLI
class DataPipeline(PlantDiseaseDataPipeline):
    """Compatibility wrapper around PlantDiseaseDataPipeline.

    Exposes prepare_datasets(data_path) used by existing callers and
    delegates to the plant-disease-aware implementation.
    """

    def prepare_datasets(self, data_path: str) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        return self.prepare_plant_disease_datasets(data_path)