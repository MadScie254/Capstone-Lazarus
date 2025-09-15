"""
Test Configuration and Utilities for CAPSTONE-LAZARUS
====================================================

Common test utilities and configuration for the test suite.
"""

import unittest
import tempfile
import shutil
import os
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import json


class BaseTestCase(unittest.TestCase):
    """Base test case with common setup and utilities"""
    
    def setUp(self):
        """Set up common test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_data_dir = Path(self.temp_dir) / 'test_data'
        self.test_data_dir.mkdir(exist_ok=True)
        
        # Set up test environment variables
        os.environ['TESTING'] = 'true'
        os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disable GPU for tests
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        os.environ.pop('TESTING', None)
        os.environ.pop('CUDA_VISIBLE_DEVICES', None)
    
    def create_mock_image_data(self, shape=(224, 224, 3), batch_size=1):
        """Create mock image data for testing"""
        if batch_size == 1:
            return np.random.rand(*shape).astype(np.float32)
        else:
            return np.random.rand(batch_size, *shape).astype(np.float32)
    
    def create_mock_labels(self, num_samples, num_classes=10):
        """Create mock labels for testing"""
        return np.random.randint(0, num_classes, size=num_samples)
    
    def create_mock_dataset_structure(self, base_path=None, classes=None, samples_per_class=5):
        """Create mock dataset directory structure"""
        if base_path is None:
            base_path = self.test_data_dir
        
        if classes is None:
            classes = ['class_0', 'class_1', 'class_2']
        
        # Create directory structure
        for split in ['train', 'val', 'test']:
            split_dir = Path(base_path) / split
            split_dir.mkdir(exist_ok=True)
            
            for class_name in classes:
                class_dir = split_dir / class_name
                class_dir.mkdir(exist_ok=True)
                
                # Create mock image files
                for i in range(samples_per_class):
                    mock_image = class_dir / f'image_{i:03d}.jpg'
                    # Create a minimal JPEG-like file
                    with open(mock_image, 'wb') as f:
                        f.write(b'\xff\xd8\xff\xe0')  # JPEG header
        
        return base_path
    
    def create_mock_config_file(self, config_data, filename='test_config.json'):
        """Create a mock configuration file"""
        config_path = self.test_data_dir / filename
        
        with open(config_path, 'w') as f:
            if filename.endswith('.json'):
                json.dump(config_data, f, indent=2)
            elif filename.endswith(('.yaml', '.yml')):
                import yaml
                yaml.dump(config_data, f)
        
        return config_path
    
    def assert_file_exists(self, file_path):
        """Assert that a file exists"""
        self.assertTrue(Path(file_path).exists(), f"File does not exist: {file_path}")
    
    def assert_directory_structure(self, base_path, expected_structure):
        """Assert that directory structure matches expected"""
        base_path = Path(base_path)
        
        for item in expected_structure:
            if isinstance(item, str):
                # File or directory
                item_path = base_path / item
                self.assertTrue(item_path.exists(), f"Missing: {item_path}")
            elif isinstance(item, dict):
                # Directory with subdirectories
                for dir_name, sub_items in item.items():
                    dir_path = base_path / dir_name
                    self.assertTrue(dir_path.is_dir(), f"Missing directory: {dir_path}")
                    
                    if sub_items:
                        self.assert_directory_structure(dir_path, sub_items)


class MockTensorFlowComponents:
    """Mock TensorFlow components for testing"""
    
    @staticmethod
    def mock_model():
        """Create a mock TensorFlow model"""
        model = MagicMock()
        model.predict.return_value = np.random.rand(1, 10)
        model.fit.return_value.history = {
            'loss': [0.5, 0.3, 0.2],
            'accuracy': [0.8, 0.9, 0.95],
            'val_loss': [0.6, 0.4, 0.3],
            'val_accuracy': [0.75, 0.85, 0.9]
        }
        model.evaluate.return_value = [0.2, 0.95]  # [loss, accuracy]
        model.save = MagicMock()
        model.count_params.return_value = 1000000
        return model
    
    @staticmethod
    def mock_dataset():
        """Create a mock TensorFlow dataset"""
        dataset = MagicMock()
        dataset.batch.return_value = dataset
        dataset.prefetch.return_value = dataset
        dataset.map.return_value = dataset
        dataset.shuffle.return_value = dataset
        dataset.cardinality.return_value.numpy.return_value = 100
        dataset.class_names = ['class_0', 'class_1', 'class_2']
        return dataset
    
    @staticmethod
    def mock_keras_callback():
        """Create a mock Keras callback"""
        callback = MagicMock()
        callback.on_epoch_end = MagicMock()
        callback.on_train_end = MagicMock()
        return callback


class TestDataGenerator:
    """Generate test data for various scenarios"""
    
    @staticmethod
    def generate_classification_data(num_samples=100, num_features=10, num_classes=3, random_state=42):
        """Generate mock classification data"""
        np.random.seed(random_state)
        
        X = np.random.randn(num_samples, num_features)
        y = np.random.randint(0, num_classes, size=num_samples)
        
        return X, y
    
    @staticmethod
    def generate_image_data(num_samples=50, image_shape=(224, 224, 3), num_classes=3, random_state=42):
        """Generate mock image data"""
        np.random.seed(random_state)
        
        X = np.random.rand(num_samples, *image_shape).astype(np.float32)
        y = np.random.randint(0, num_classes, size=num_samples)
        
        return X, y
    
    @staticmethod
    def generate_time_series_data(num_samples=200, sequence_length=50, num_features=5, random_state=42):
        """Generate mock time series data"""
        np.random.seed(random_state)
        
        X = np.random.randn(num_samples, sequence_length, num_features)
        y = np.random.randn(num_samples, 1)  # Regression target
        
        return X, y


class TestUtils:
    """General testing utilities"""
    
    @staticmethod
    def patch_tensorflow():
        """Create patches for TensorFlow components"""
        patches = {
            'model': patch('tensorflow.keras.models.Sequential', return_value=MockTensorFlowComponents.mock_model()),
            'dataset': patch('tensorflow.keras.utils.image_dataset_from_directory', return_value=MockTensorFlowComponents.mock_dataset()),
            'optimizer': patch('tensorflow.keras.optimizers.Adam'),
            'loss': patch('tensorflow.keras.losses.SparseCategoricalCrossentropy'),
            'callbacks': patch('tensorflow.keras.callbacks.EarlyStopping', return_value=MockTensorFlowComponents.mock_keras_callback())
        }
        return patches
    
    @staticmethod
    def patch_mlflow():
        """Create patches for MLflow components"""
        patches = {
            'start_run': patch('mlflow.start_run'),
            'log_metric': patch('mlflow.log_metric'),
            'log_param': patch('mlflow.log_param'),
            'log_artifact': patch('mlflow.log_artifact'),
            'set_experiment': patch('mlflow.set_experiment')
        }
        return patches
    
    @staticmethod
    def assert_approximately_equal(test_case, actual, expected, tolerance=1e-6):
        """Assert that two values are approximately equal"""
        if isinstance(actual, np.ndarray) and isinstance(expected, np.ndarray):
            test_case.assertTrue(np.allclose(actual, expected, atol=tolerance))
        else:
            test_case.assertAlmostEqual(actual, expected, delta=tolerance)
    
    @staticmethod
    def capture_logs(logger_name='test_logger'):
        """Context manager to capture log messages"""
        import logging
        from io import StringIO
        
        class LogCapture:
            def __init__(self, logger_name):
                self.logger = logging.getLogger(logger_name)
                self.stream = StringIO()
                self.handler = logging.StreamHandler(self.stream)
                self.original_level = self.logger.level
                
            def __enter__(self):
                self.logger.addHandler(self.handler)
                self.logger.setLevel(logging.DEBUG)
                return self.stream
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                self.logger.removeHandler(self.handler)
                self.logger.setLevel(self.original_level)
        
        return LogCapture(logger_name)


# Test discovery and runner utilities
def discover_tests(start_dir='.', pattern='test*.py'):
    """Discover and return test suite"""
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir, pattern=pattern, top_level_dir=None)
    return suite


def run_test_suite(test_suite=None, verbosity=2):
    """Run test suite with specified verbosity"""
    if test_suite is None:
        test_suite = discover_tests()
    
    runner = unittest.TextTestRunner(verbosity=verbosity, buffer=True)
    result = runner.run(test_suite)
    
    return result


# Custom test decorators
def skip_if_no_gpu(test_func):
    """Skip test if no GPU is available"""
    def wrapper(*args, **kwargs):
        try:
            import tensorflow as tf
            if not tf.config.list_physical_devices('GPU'):
                unittest.skip("No GPU available")(test_func)(*args, **kwargs)
            else:
                test_func(*args, **kwargs)
        except ImportError:
            unittest.skip("TensorFlow not available")(test_func)(*args, **kwargs)
    
    return wrapper


def requires_internet(test_func):
    """Skip test if no internet connection"""
    import socket
    
    def wrapper(*args, **kwargs):
        try:
            # Try to connect to a reliable server
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            test_func(*args, **kwargs)
        except OSError:
            unittest.skip("No internet connection")(test_func)(*args, **kwargs)
    
    return wrapper


# Test configuration
TEST_CONFIG = {
    'data_dir': 'tests/test_data',
    'fixtures_dir': 'tests/fixtures',
    'timeout': 30,  # seconds
    'random_seed': 42,
    'small_dataset_size': 50,
    'medium_dataset_size': 500,
    'large_dataset_size': 5000
}


if __name__ == '__main__':
    # Run all tests when this module is executed
    test_suite = discover_tests()
    result = run_test_suite(test_suite)
    
    # Exit with appropriate code
    exit(0 if result.wasSuccessful() else 1)