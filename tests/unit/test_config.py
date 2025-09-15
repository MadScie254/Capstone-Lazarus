"""
Unit Tests for CAPSTONE-LAZARUS Core Configuration
=================================================

Tests for the configuration system and core utilities.
"""

import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add src to Python path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from config.settings import ModelConfig, TrainingConfig, DataConfig, Config


class TestModelConfig(unittest.TestCase):
    """Test ModelConfig dataclass"""
    
    def test_default_values(self):
        """Test default configuration values"""
        config = ModelConfig()
        
        self.assertEqual(config.name, 'resnet50')
        self.assertEqual(config.num_classes, 10)
        self.assertEqual(config.input_shape, (224, 224, 3))
        self.assertEqual(config.dropout_rate, 0.2)
        self.assertIsInstance(config.pretrained, bool)
    
    def test_custom_values(self):
        """Test custom configuration values"""
        config = ModelConfig(
            name='efficientnet',
            num_classes=5,
            input_shape=(256, 256, 3),
            dropout_rate=0.1
        )
        
        self.assertEqual(config.name, 'efficientnet')
        self.assertEqual(config.num_classes, 5)
        self.assertEqual(config.input_shape, (256, 256, 3))
        self.assertEqual(config.dropout_rate, 0.1)
    
    def test_validation(self):
        """Test configuration validation"""
        # Valid configurations should not raise errors
        ModelConfig(num_classes=1)
        ModelConfig(num_classes=1000)
        
        # Test edge cases
        with self.assertRaises(ValueError):
            ModelConfig(num_classes=0)
        
        with self.assertRaises(ValueError):
            ModelConfig(dropout_rate=-0.1)
        
        with self.assertRaises(ValueError):
            ModelConfig(dropout_rate=1.1)


class TestTrainingConfig(unittest.TestCase):
    """Test TrainingConfig dataclass"""
    
    def test_default_values(self):
        """Test default training configuration"""
        config = TrainingConfig()
        
        self.assertEqual(config.batch_size, 32)
        self.assertEqual(config.epochs, 100)
        self.assertEqual(config.learning_rate, 0.001)
        self.assertEqual(config.optimizer, 'adam')
        self.assertIn('accuracy', config.metrics)
    
    def test_custom_values(self):
        """Test custom training configuration"""
        config = TrainingConfig(
            batch_size=64,
            epochs=50,
            learning_rate=0.01,
            optimizer='sgd'
        )
        
        self.assertEqual(config.batch_size, 64)
        self.assertEqual(config.epochs, 50)
        self.assertEqual(config.learning_rate, 0.01)
        self.assertEqual(config.optimizer, 'sgd')
    
    def test_validation(self):
        """Test training configuration validation"""
        # Valid configurations
        TrainingConfig(batch_size=1, epochs=1, learning_rate=0.0001)
        
        # Invalid configurations
        with self.assertRaises(ValueError):
            TrainingConfig(batch_size=0)
        
        with self.assertRaises(ValueError):
            TrainingConfig(epochs=0)
        
        with self.assertRaises(ValueError):
            TrainingConfig(learning_rate=0)


class TestDataConfig(unittest.TestCase):
    """Test DataConfig dataclass"""
    
    def test_default_values(self):
        """Test default data configuration"""
        config = DataConfig()
        
        self.assertEqual(config.train_path, 'data/train')
        self.assertEqual(config.val_path, 'data/val')
        self.assertEqual(config.batch_size, 32)
        self.assertTrue(config.shuffle)
    
    def test_custom_paths(self):
        """Test custom data paths"""
        config = DataConfig(
            train_path='/custom/train',
            val_path='/custom/val',
            test_path='/custom/test'
        )
        
        self.assertEqual(config.train_path, '/custom/train')
        self.assertEqual(config.val_path, '/custom/val')
        self.assertEqual(config.test_path, '/custom/test')
    
    def test_augmentation_config(self):
        """Test data augmentation configuration"""
        config = DataConfig(
            augmentation=True,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True
        )
        
        self.assertTrue(config.augmentation)
        self.assertEqual(config.rotation_range, 20)
        self.assertEqual(config.width_shift_range, 0.2)
        self.assertTrue(config.horizontal_flip)


class TestConfig(unittest.TestCase):
    """Test main Config class"""
    
    def test_default_config(self):
        """Test default configuration creation"""
        config = Config()
        
        self.assertIsInstance(config.model, ModelConfig)
        self.assertIsInstance(config.training, TrainingConfig)
        self.assertIsInstance(config.data, DataConfig)
    
    def test_config_from_dict(self):
        """Test configuration creation from dictionary"""
        config_dict = {
            'model': {
                'name': 'vgg16',
                'num_classes': 5
            },
            'training': {
                'batch_size': 16,
                'epochs': 20
            },
            'data': {
                'train_path': '/data/custom/train'
            }
        }
        
        config = Config.from_dict(config_dict)
        
        self.assertEqual(config.model.name, 'vgg16')
        self.assertEqual(config.model.num_classes, 5)
        self.assertEqual(config.training.batch_size, 16)
        self.assertEqual(config.training.epochs, 20)
        self.assertEqual(config.data.train_path, '/data/custom/train')
    
    def test_config_to_dict(self):
        """Test configuration conversion to dictionary"""
        config = Config(
            model=ModelConfig(name='custom', num_classes=3),
            training=TrainingConfig(batch_size=8)
        )
        
        config_dict = config.to_dict()
        
        self.assertEqual(config_dict['model']['name'], 'custom')
        self.assertEqual(config_dict['model']['num_classes'], 3)
        self.assertEqual(config_dict['training']['batch_size'], 8)
    
    def test_config_save_load(self):
        """Test configuration save/load functionality"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / 'test_config.json'
            
            # Create and save config
            original_config = Config(
                model=ModelConfig(name='test_model', num_classes=7),
                training=TrainingConfig(batch_size=16, epochs=5)
            )
            original_config.save(config_path)
            
            # Load config
            loaded_config = Config.load(config_path)
            
            # Verify loaded config
            self.assertEqual(loaded_config.model.name, 'test_model')
            self.assertEqual(loaded_config.model.num_classes, 7)
            self.assertEqual(loaded_config.training.batch_size, 16)
            self.assertEqual(loaded_config.training.epochs, 5)
    
    def test_config_validation(self):
        """Test full configuration validation"""
        # Valid configuration should not raise errors
        config = Config(
            model=ModelConfig(name='valid', num_classes=10),
            training=TrainingConfig(batch_size=32, epochs=10, learning_rate=0.001)
        )
        config.validate()
        
        # Test cross-field validation
        config_dict = {
            'model': {'num_classes': 10},
            'training': {'batch_size': 32},
            'data': {'batch_size': 64}  # Different batch size should trigger warning
        }
        
        config = Config.from_dict(config_dict)
        # This should work but might log a warning
        config.validate()
    
    def test_environment_variable_substitution(self):
        """Test environment variable substitution in config"""
        with patch.dict('os.environ', {'TEST_BATCH_SIZE': '64', 'TEST_LR': '0.01'}):
            config_dict = {
                'training': {
                    'batch_size': '${TEST_BATCH_SIZE}',
                    'learning_rate': '${TEST_LR}'
                }
            }
            
            config = Config.from_dict(config_dict)
            
            # Environment variables should be substituted
            self.assertEqual(config.training.batch_size, 64)
            self.assertEqual(config.training.learning_rate, 0.01)


if __name__ == '__main__':
    unittest.main(verbosity=2)