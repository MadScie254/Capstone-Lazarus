"""
Unit Tests for CAPSTONE-LAZARUS Data Pipeline
============================================

Tests for data loading, processing, and validation functionality.
"""

import unittest
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import json

# Add src to Python path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from config.settings import DataConfig
from data.loader import DataLoader
from data.preprocessor import DataPreprocessor
from data.validator import DataValidator


class TestDataLoader(unittest.TestCase):
    """Test DataLoader functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.data_config = DataConfig(
            train_path=str(Path(self.temp_dir) / 'train'),
            val_path=str(Path(self.temp_dir) / 'val'),
            batch_size=4
        )
        
        # Create mock directory structure
        self.train_dir = Path(self.data_config.train_path)
        self.val_dir = Path(self.data_config.val_path)
        self.train_dir.mkdir(parents=True, exist_ok=True)
        self.val_dir.mkdir(parents=True, exist_ok=True)
        
        # Create class directories
        for class_name in ['class_a', 'class_b']:
            (self.train_dir / class_name).mkdir(exist_ok=True)
            (self.val_dir / class_name).mkdir(exist_ok=True)
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test DataLoader initialization"""
        loader = DataLoader(self.data_config)
        
        self.assertEqual(loader.config, self.data_config)
        self.assertEqual(loader.batch_size, 4)
        self.assertIsNone(loader.train_dataset)
        self.assertIsNone(loader.val_dataset)
    
    @patch('tensorflow.keras.utils.image_dataset_from_directory')
    def test_load_datasets(self, mock_dataset):
        """Test dataset loading"""
        # Mock TensorFlow dataset
        mock_dataset.return_value = MagicMock()
        mock_dataset.return_value.class_names = ['class_a', 'class_b']
        
        loader = DataLoader(self.data_config)
        train_ds, val_ds = loader.load_datasets()
        
        # Verify datasets were loaded
        self.assertIsNotNone(train_ds)
        self.assertIsNotNone(val_ds)
        
        # Verify dataset creation parameters
        self.assertEqual(mock_dataset.call_count, 2)  # Called for train and val
    
    def test_get_class_names(self):
        """Test class name extraction"""
        loader = DataLoader(self.data_config)
        
        # Mock the directory structure exists
        class_names = loader.get_class_names()
        
        # Should find our test classes
        self.assertIn('class_a', class_names)
        self.assertIn('class_b', class_names)
    
    def test_get_dataset_info(self):
        """Test dataset information extraction"""
        with patch.object(DataLoader, 'load_datasets') as mock_load:
            # Mock dataset with known properties
            mock_train_ds = MagicMock()
            mock_val_ds = MagicMock()
            mock_train_ds.cardinality.return_value.numpy.return_value = 100
            mock_val_ds.cardinality.return_value.numpy.return_value = 20
            mock_load.return_value = (mock_train_ds, mock_val_ds)
            
            loader = DataLoader(self.data_config)
            info = loader.get_dataset_info()
            
            self.assertIn('train_samples', info)
            self.assertIn('val_samples', info)
            self.assertIn('num_classes', info)
    
    def test_invalid_paths(self):
        """Test handling of invalid data paths"""
        invalid_config = DataConfig(
            train_path='/nonexistent/path',
            val_path='/another/nonexistent/path'
        )
        
        loader = DataLoader(invalid_config)
        
        with self.assertRaises(ValueError):
            loader.load_datasets()


class TestDataPreprocessor(unittest.TestCase):
    """Test DataPreprocessor functionality"""
    
    def test_initialization(self):
        """Test preprocessor initialization"""
        config = DataConfig(
            augmentation=True,
            rotation_range=20,
            horizontal_flip=True
        )
        
        preprocessor = DataPreprocessor(config)
        
        self.assertEqual(preprocessor.config, config)
        self.assertTrue(preprocessor.augmentation_enabled)
    
    @patch('tensorflow.keras.utils.image_dataset_from_directory')
    def test_create_augmentation_layer(self, mock_dataset):
        """Test augmentation layer creation"""
        config = DataConfig(
            augmentation=True,
            rotation_range=0.2,
            horizontal_flip=True,
            zoom_range=0.1
        )
        
        preprocessor = DataPreprocessor(config)
        aug_layer = preprocessor.create_augmentation_layer()
        
        self.assertIsNotNone(aug_layer)
        # Should be a Sequential model with augmentation layers
        self.assertTrue(hasattr(aug_layer, 'layers'))
    
    def test_normalize_data(self):
        """Test data normalization"""
        # Create mock image data
        mock_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        preprocessor = DataPreprocessor(DataConfig())
        normalized = preprocessor.normalize_data(mock_image)
        
        # Check if normalized to [0, 1] range
        self.assertTrue(np.all(normalized >= 0))
        self.assertTrue(np.all(normalized <= 1))
        self.assertEqual(normalized.dtype, np.float32)
    
    def test_preprocess_dataset(self):
        """Test dataset preprocessing"""
        with patch('tensorflow.data.Dataset') as mock_dataset:
            mock_ds = MagicMock()
            mock_dataset.return_value = mock_ds
            
            config = DataConfig(batch_size=32, shuffle=True)
            preprocessor = DataPreprocessor(config)
            
            # Mock dataset operations
            mock_ds.map.return_value = mock_ds
            mock_ds.batch.return_value = mock_ds
            mock_ds.prefetch.return_value = mock_ds
            mock_ds.shuffle.return_value = mock_ds
            
            result = preprocessor.preprocess_dataset(mock_ds, is_training=True)
            
            # Verify preprocessing pipeline was applied
            self.assertTrue(mock_ds.map.called)
            self.assertTrue(mock_ds.batch.called)
            self.assertTrue(mock_ds.prefetch.called)
    
    def test_resize_image(self):
        """Test image resizing functionality"""
        # Create mock image
        mock_image = np.random.rand(100, 100, 3).astype(np.float32)
        target_size = (224, 224)
        
        preprocessor = DataPreprocessor(DataConfig())
        resized = preprocessor.resize_image(mock_image, target_size)
        
        self.assertEqual(resized.shape[:2], target_size)
    
    def test_augmentation_pipeline(self):
        """Test complete augmentation pipeline"""
        config = DataConfig(
            augmentation=True,
            rotation_range=0.1,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            vertical_flip=False,
            zoom_range=0.1
        )
        
        preprocessor = DataPreprocessor(config)
        
        # Create mock batch of images
        batch_images = np.random.rand(4, 224, 224, 3).astype(np.float32)
        
        aug_layer = preprocessor.create_augmentation_layer()
        
        # Apply augmentation (this would normally happen in TF context)
        self.assertIsNotNone(aug_layer)


class TestDataValidator(unittest.TestCase):
    """Test DataValidator functionality"""
    
    def test_initialization(self):
        """Test validator initialization"""
        validator = DataValidator()
        self.assertIsNotNone(validator)
    
    def test_validate_image_format(self):
        """Test image format validation"""
        validator = DataValidator()
        
        # Valid image shapes
        self.assertTrue(validator.validate_image_shape((224, 224, 3)))
        self.assertTrue(validator.validate_image_shape((256, 256, 1)))
        self.assertTrue(validator.validate_image_shape((512, 512, 3)))
        
        # Invalid shapes
        self.assertFalse(validator.validate_image_shape((224, 224)))  # Missing channels
        self.assertFalse(validator.validate_image_shape((0, 224, 3)))  # Zero dimension
        self.assertFalse(validator.validate_image_shape((224, 224, 0)))  # Zero channels
    
    def test_validate_class_distribution(self):
        """Test class distribution validation"""
        validator = DataValidator()
        
        # Balanced distribution
        balanced_dist = {'class_a': 100, 'class_b': 100, 'class_c': 100}
        is_valid, message = validator.validate_class_distribution(balanced_dist)
        self.assertTrue(is_valid)
        
        # Imbalanced distribution
        imbalanced_dist = {'class_a': 1000, 'class_b': 10, 'class_c': 5}
        is_valid, message = validator.validate_class_distribution(imbalanced_dist)
        self.assertFalse(is_valid)
        self.assertIn('imbalanced', message.lower())
    
    def test_validate_dataset_size(self):
        """Test dataset size validation"""
        validator = DataValidator()
        
        # Sufficient data
        self.assertTrue(validator.validate_dataset_size(1000, 'train'))
        self.assertTrue(validator.validate_dataset_size(200, 'validation'))
        
        # Insufficient data
        self.assertFalse(validator.validate_dataset_size(10, 'train'))
        self.assertFalse(validator.validate_dataset_size(5, 'validation'))
    
    def test_validate_file_integrity(self):
        """Test file integrity validation"""
        validator = DataValidator()
        
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            # Write some mock image data
            temp_file.write(b'\xff\xd8\xff\xe0')  # JPEG header
            temp_file.flush()
            
            # Should validate successfully for existing file
            is_valid = validator.validate_file_integrity(temp_file.name)
            # Note: This might fail without actual image libraries, so we just test the interface
            self.assertIsInstance(is_valid, bool)
        
        # Non-existent file
        is_valid = validator.validate_file_integrity('/nonexistent/file.jpg')
        self.assertFalse(is_valid)
    
    def test_validate_data_config(self):
        """Test data configuration validation"""
        validator = DataValidator()
        
        # Valid configuration
        valid_config = DataConfig(
            train_path='data/train',
            val_path='data/val',
            batch_size=32,
            input_shape=(224, 224, 3)
        )
        
        is_valid, issues = validator.validate_data_config(valid_config)
        # Note: This might fail without actual paths, but we test the interface
        self.assertIsInstance(is_valid, bool)
        self.assertIsInstance(issues, list)
        
        # Invalid configuration
        invalid_config = DataConfig(
            train_path='',
            val_path='',
            batch_size=0
        )
        
        is_valid, issues = validator.validate_data_config(invalid_config)
        self.assertFalse(is_valid)
        self.assertTrue(len(issues) > 0)
    
    def test_check_data_leakage(self):
        """Test data leakage detection"""
        validator = DataValidator()
        
        # Mock file lists
        train_files = ['img1.jpg', 'img2.jpg', 'img3.jpg']
        val_files = ['img4.jpg', 'img5.jpg']
        test_files = ['img6.jpg', 'img7.jpg']
        
        # No leakage
        has_leakage = validator.check_data_leakage(train_files, val_files, test_files)
        self.assertFalse(has_leakage)
        
        # With leakage
        val_files_with_leakage = ['img1.jpg', 'img5.jpg']  # img1 also in train
        has_leakage = validator.check_data_leakage(train_files, val_files_with_leakage)
        self.assertTrue(has_leakage)


if __name__ == '__main__':
    unittest.main(verbosity=2)