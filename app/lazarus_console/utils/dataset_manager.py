"""
Dataset Management Utilities for Lazarus Console
Handles dataset loading, analysis, and manifest management
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import json
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import streamlit as st

class DatasetManager:
    """Centralized dataset management and analysis"""
    
    def __init__(self, project_root: Optional[Path] = None):
        """Initialize dataset manager with real project paths"""
        if project_root is None:
            # Default to current directory structure
            self.project_root = Path(".")
        else:
            self.project_root = Path(project_root)
            
        self.data_dir = self.project_root / 'data'
        self.features_dir = self.project_root / 'features'
        self.models_dir = self.project_root / 'models'
        self.manifest_file = self.features_dir / 'manifest_features.v001.csv'
        self.model_registry_path = self.models_dir / 'model_registry.json'
        self.class_names_path = self.models_dir / 'class_names.json'
        
        # Load real class names if available
        self.class_names = self._load_class_names()
        
        # Initialize data structures
        self._initialize_data()
    
    def _load_class_names(self) -> List[str]:
        """Load class names from the real model registry"""
        if self.class_names_path.exists():
            try:
                with open(self.class_names_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load class names: {e}")
        
        # Fallback: scan data directory for class folders
        if self.data_dir.exists():
            return [d.name for d in self.data_dir.iterdir() if d.is_dir()]
        
        return []
    
    def _initialize_data(self):
        """Initialize data structures after class names are loaded"""
        # Cached data
        self._manifest_cache = None
        self._class_stats_cache = None
        self._real_dataset_stats = None
        self._image_stats_cache = None
        
        # Load or create manifest
        self.manifest = self._load_or_create_manifest()
    
    def _load_or_create_manifest(self) -> Optional[pd.DataFrame]:
        """Load existing manifest or create new one"""
        try:
            if self.manifest_file.exists():
                manifest = pd.read_csv(self.manifest_file)
                self._manifest_cache = manifest
                return manifest
            else:
                # Create manifest from data directory
                return self._create_manifest_from_data()
        except Exception as e:
            print(f"Error loading/creating manifest: {e}")
            return None
    
    def _create_manifest_from_data(self) -> Optional[pd.DataFrame]:
        """Create manifest by scanning data directory"""
        try:
            if not self.data_dir.exists():
                print(f"Data directory not found: {self.data_dir}")
                return None
            
            data_rows = []
            image_id = 0
            
            for class_dir in self.data_dir.iterdir():
                if not class_dir.is_dir():
                    continue
                
                class_name = class_dir.name
                print(f"Processing class: {class_name}")
                
                # Find all image files
                image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
                image_files = []
                
                for ext in image_extensions:
                    image_files.extend(class_dir.glob(f"*{ext}"))
                
                for img_path in image_files:
                    try:
                        # Get image info
                        img_info = self._get_image_info(img_path)
                        
                        data_rows.append({
                            'image_id': f"img_{image_id:06d}",
                            'image_path': str(img_path),
                            'class_name': class_name,
                            'filename': img_path.name,
                            'width': img_info['width'],
                            'height': img_info['height'],
                            'channels': img_info['channels'],
                            'file_size': img_info['file_size'],
                            'aspect_ratio': img_info['aspect_ratio']
                        })
                        
                        image_id += 1
                        
                    except Exception as e:
                        print(f"Error processing {img_path}: {e}")
                        continue
            
            if data_rows:
                manifest = pd.DataFrame(data_rows)
                
                # Save manifest
                self.features_dir.mkdir(exist_ok=True)
                manifest.to_csv(self.manifest_file, index=False)
                
                print(f"Created manifest with {len(manifest)} images")
                self._manifest_cache = manifest
                return manifest
            else:
                print("No images found in data directory")
                return None
                
        except Exception as e:
            print(f"Error creating manifest: {e}")
            return None
    
    def _get_image_info(self, img_path: Path) -> Dict[str, Any]:
        """Get basic image information"""
        try:
            # Try with PIL first (more reliable for metadata)
            with Image.open(img_path) as img:
                width, height = img.size
                channels = len(img.getbands()) if hasattr(img, 'getbands') else 3
            
            file_size = img_path.stat().st_size
            aspect_ratio = width / height if height > 0 else 1.0
            
            return {
                'width': width,
                'height': height,
                'channels': channels,
                'file_size': file_size,
                'aspect_ratio': aspect_ratio
            }
            
        except Exception as e:
            # Fallback to OpenCV
            try:
                img = cv2.imread(str(img_path))
                height, width, channels = img.shape
                file_size = img_path.stat().st_size
                aspect_ratio = width / height if height > 0 else 1.0
                
                return {
                    'width': width,
                    'height': height,
                    'channels': channels,
                    'file_size': file_size,
                    'aspect_ratio': aspect_ratio
                }
            except Exception as e2:
                print(f"Error getting image info for {img_path}: {e2}")
                return {
                    'width': 224,
                    'height': 224,
                    'channels': 3,
                    'file_size': 0,
                    'aspect_ratio': 1.0
                }
    
    def get_manifest(self) -> Optional[pd.DataFrame]:
        """Get dataset manifest"""
        return self.manifest
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get class distribution counts"""
        if self.manifest is None:
            return {}
        
        return dict(self.manifest['class_name'].value_counts())
    
    def get_class_statistics(self) -> Dict[str, Any]:
        """Get detailed class statistics"""
        if self._class_stats_cache is not None:
            return self._class_stats_cache
        
        if self.manifest is None:
            return {}
        
        stats = {}
        class_counts = self.manifest['class_name'].value_counts()
        
        stats['total_images'] = len(self.manifest)
        stats['num_classes'] = len(class_counts)
        stats['class_counts'] = dict(class_counts)
        stats['min_samples'] = class_counts.min()
        stats['max_samples'] = class_counts.max()
        stats['mean_samples'] = class_counts.mean()
        stats['std_samples'] = class_counts.std()
        
        # Imbalance metrics
        stats['imbalance_ratio'] = stats['max_samples'] / stats['min_samples'] if stats['min_samples'] > 0 else float('inf')
        stats['gini_coefficient'] = self._calculate_gini_coefficient(class_counts.values)
        
        # Identify problematic classes
        median_count = class_counts.median()
        stats['small_classes'] = class_counts[class_counts < median_count * 0.5].index.tolist()
        stats['large_classes'] = class_counts[class_counts > median_count * 2.0].index.tolist()
        
        self._class_stats_cache = stats
        return stats
    
    def get_image_statistics(self) -> Dict[str, Any]:
        """Get detailed image statistics"""
        if self._image_stats_cache is not None:
            return self._image_stats_cache
        
        if self.manifest is None:
            return {}
        
        stats = {}
        
        # Resolution statistics
        stats['width_stats'] = {
            'min': self.manifest['width'].min(),
            'max': self.manifest['width'].max(),
            'mean': self.manifest['width'].mean(),
            'std': self.manifest['width'].std()
        }
        
        stats['height_stats'] = {
            'min': self.manifest['height'].min(),
            'max': self.manifest['height'].max(),
            'mean': self.manifest['height'].mean(),
            'std': self.manifest['height'].std()
        }
        
        # Aspect ratio statistics
        stats['aspect_ratio_stats'] = {
            'min': self.manifest['aspect_ratio'].min(),
            'max': self.manifest['aspect_ratio'].max(),
            'mean': self.manifest['aspect_ratio'].mean(),
            'std': self.manifest['aspect_ratio'].std()
        }
        
        # File size statistics
        stats['file_size_stats'] = {
            'min_mb': self.manifest['file_size'].min() / (1024 * 1024),
            'max_mb': self.manifest['file_size'].max() / (1024 * 1024),
            'mean_mb': self.manifest['file_size'].mean() / (1024 * 1024),
            'total_gb': self.manifest['file_size'].sum() / (1024 * 1024 * 1024)
        }
        
        # Resolution distribution
        resolution_counts = self.manifest.groupby(['width', 'height']).size()
        stats['common_resolutions'] = dict(resolution_counts.nlargest(10))
        
        # Channel distribution
        stats['channel_distribution'] = dict(self.manifest['channels'].value_counts())
        
        self._image_stats_cache = stats
        return stats
    
    def _calculate_gini_coefficient(self, values) -> float:
        """Calculate Gini coefficient for class imbalance"""
        try:
            values = np.array(values)
            values = values[values > 0]  # Remove zeros
            n = len(values)
            
            if n == 0:
                return 0.0
            
            # Sort values
            sorted_values = np.sort(values)
            
            # Calculate Gini coefficient
            index = np.arange(1, n + 1)
            gini = (2 * np.sum(index * sorted_values)) / (n * np.sum(sorted_values)) - (n + 1) / n
            
            return gini
        except:
            return 0.0
    
    def get_sample_images(self, class_name: str = None, num_samples: int = 5) -> List[Dict[str, Any]]:
        """Get sample images for visualization"""
        if self.manifest is None:
            return []
        
        try:
            if class_name:
                # Get samples from specific class
                class_samples = self.manifest[self.manifest['class_name'] == class_name]
                if len(class_samples) == 0:
                    return []
                samples = class_samples.sample(min(num_samples, len(class_samples)))
            else:
                # Get samples from all classes
                samples = self.manifest.sample(min(num_samples, len(self.manifest)))
            
            sample_info = []
            for _, row in samples.iterrows():
                try:
                    img_path = Path(row['image_path'])
                    if img_path.exists():
                        sample_info.append({
                            'image_id': row['image_id'],
                            'path': str(img_path),
                            'class_name': row['class_name'],
                            'filename': row['filename'],
                            'width': row['width'],
                            'height': row['height'],
                            'size_mb': row['file_size'] / (1024 * 1024)
                        })
                except Exception as e:
                    print(f"Error processing sample {row['image_id']}: {e}")
                    continue
            
            return sample_info
            
        except Exception as e:
            print(f"Error getting sample images: {e}")
            return []
    
    def analyze_class_balance(self) -> Dict[str, Any]:
        """Analyze class balance and suggest improvements"""
        class_stats = self.get_class_statistics()
        
        if not class_stats:
            return {'error': 'No class statistics available'}
        
        analysis = {
            'balance_status': 'balanced',
            'recommendations': [],
            'imbalance_severity': 'low',
            'suggested_actions': []
        }
        
        # Assess balance
        imbalance_ratio = class_stats.get('imbalance_ratio', 1.0)
        gini_coeff = class_stats.get('gini_coefficient', 0.0)
        
        if imbalance_ratio > 10 or gini_coeff > 0.4:
            analysis['balance_status'] = 'severely_imbalanced'
            analysis['imbalance_severity'] = 'high'
        elif imbalance_ratio > 3 or gini_coeff > 0.2:
            analysis['balance_status'] = 'moderately_imbalanced'
            analysis['imbalance_severity'] = 'medium'
        
        # Generate recommendations
        if analysis['imbalance_severity'] != 'low':
            small_classes = class_stats.get('small_classes', [])
            large_classes = class_stats.get('large_classes', [])
            
            if small_classes:
                analysis['recommendations'].append(f"Consider data augmentation for: {', '.join(small_classes[:3])}")
                analysis['suggested_actions'].append('data_augmentation')
            
            if large_classes:
                analysis['recommendations'].append(f"Consider undersampling for: {', '.join(large_classes[:3])}")
                analysis['suggested_actions'].append('undersampling')
            
            analysis['recommendations'].append("Use class weights during training")
            analysis['suggested_actions'].append('class_weighting')
            
            if imbalance_ratio > 5:
                analysis['recommendations'].append("Consider focal loss for severe imbalance")
                analysis['suggested_actions'].append('focal_loss')
        
        return analysis
    
    def get_augmentation_suggestions(self, class_name: str) -> List[str]:
        """Get data augmentation suggestions for a specific class"""
        suggestions = [
            "Horizontal flip (mirror images)",
            "Rotation (±15 degrees)",
            "Brightness adjustment (±20%)",
            "Contrast adjustment (±20%)",
            "Color jittering",
            "Random crop and resize",
            "Gaussian noise addition",
            "Elastic deformation"
        ]
        
        # Customize based on class name
        if 'leaf' in class_name.lower() or 'spot' in class_name.lower():
            suggestions.extend([
                "Random erasing (simulate occlusion)",
                "Cutout augmentation",
                "MixUp with similar classes"
            ])
        
        return suggestions
    
    def refresh_manifest(self):
        """Refresh manifest by rescanning data directory"""
        self._manifest_cache = None
        self._class_stats_cache = None
        self._image_stats_cache = None
        self.manifest = self._create_manifest_from_data()
    
    def export_analysis_report(self) -> Dict[str, Any]:
        """Export comprehensive dataset analysis report"""
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'dataset_overview': {
                'data_directory': str(self.data_dir),
                'manifest_file': str(self.manifest_file),
                'total_images': len(self.manifest) if self.manifest is not None else 0
            },
            'class_statistics': self.get_class_statistics(),
            'image_statistics': self.get_image_statistics(),
            'balance_analysis': self.analyze_class_balance()
        }
        
        return report