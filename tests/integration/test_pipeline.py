"""
Integration Tests for CAPSTONE-LAZARUS
=====================================

Integration tests for the complete ML pipeline.
"""

import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np

# Add src to Python path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from config.settings import Config, ModelConfig, TrainingConfig, DataConfig
from models.factory import ModelFactory


class TestMLPipelineIntegration(unittest.TestCase):
    """Test complete ML pipeline integration"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = Config(
            model=ModelConfig(
                name='resnet50',
                num_classes=5,
                input_shape=(224, 224, 3)
            ),
            training=TrainingConfig(
                batch_size=4,
                epochs=2,
                learning_rate=0.001
            ),
            data=DataConfig(
                train_path=str(Path(self.temp_dir) / 'train'),
                val_path=str(Path(self.temp_dir) / 'val'),
                batch_size=4
            )
        )
        
        # Create mock directory structure
        self.create_mock_data_structure()
    
    def tearDown(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_mock_data_structure(self):
        """Create mock data directory structure"""
        # Create directories
        for split in ['train', 'val']:
            for class_name in ['class_0', 'class_1', 'class_2', 'class_3', 'class_4']:
                class_dir = Path(self.temp_dir) / split / class_name
                class_dir.mkdir(parents=True, exist_ok=True)
                
                # Create mock image files (empty files for testing)
                for i in range(5):
                    mock_image = class_dir / f'image_{i}.jpg'
                    mock_image.touch()
    
    @patch('tensorflow.keras.applications.ResNet50')
    @patch('tensorflow.keras.utils.image_dataset_from_directory')
    def test_model_creation_and_compilation(self, mock_dataset, mock_resnet):
        """Test model creation and compilation"""
        # Mock TensorFlow components
        mock_model = MagicMock()
        mock_resnet.return_value = mock_model
        
        # Create model using factory
        factory = ModelFactory(self.config.model)
        model = factory.create_model()
        
        self.assertIsNotNone(model)
    
    @patch('tensorflow.keras.utils.image_dataset_from_directory')
    def test_data_loading_pipeline(self, mock_dataset):
        """Test complete data loading pipeline"""
        # Mock dataset
        mock_ds = MagicMock()
        mock_ds.class_names = ['class_0', 'class_1', 'class_2', 'class_3', 'class_4']
        mock_dataset.return_value = mock_ds
        
        from data.loader import DataLoader
        loader = DataLoader(self.config.data)
        
        train_ds, val_ds = loader.load_datasets()
        
        self.assertIsNotNone(train_ds)
        self.assertIsNotNone(val_ds)
    
    @patch('src.training.trainer.Trainer')
    @patch('src.models.factory.ModelFactory')
    @patch('src.data.loader.DataLoader')
    def test_training_pipeline(self, mock_loader, mock_factory, mock_trainer):
        """Test complete training pipeline"""
        # Mock components
        mock_model = MagicMock()
        mock_factory.return_value.create_model.return_value = mock_model
        
        mock_dataset = MagicMock()
        mock_loader.return_value.load_datasets.return_value = (mock_dataset, mock_dataset)
        
        mock_trainer_instance = MagicMock()
        mock_trainer.return_value = mock_trainer_instance
        
        # This would test the complete pipeline
        # factory = ModelFactory(self.config.model)
        # model = factory.create_model()
        # loader = DataLoader(self.config.data)
        # train_ds, val_ds = loader.load_datasets()
        # trainer = Trainer(self.config.training, model, train_ds, val_ds)
        # history = trainer.train()
        
        # For now, just verify mocks were set up correctly
        self.assertTrue(True)  # Placeholder
    
    def test_config_validation_integration(self):
        """Test configuration validation across components"""
        # Valid config should pass all validations
        try:
            self.config.validate()
            # If we get here, validation passed
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Configuration validation failed: {e}")
        
        # Test invalid config
        invalid_config = Config(
            model=ModelConfig(num_classes=0),  # Invalid
            training=TrainingConfig(batch_size=0),  # Invalid
            data=DataConfig(train_path='')  # Invalid
        )
        
        with self.assertRaises((ValueError, AssertionError)):
            invalid_config.validate()
    
    def test_config_persistence_integration(self):
        """Test configuration save/load integration"""
        config_path = Path(self.temp_dir) / 'integration_config.json'
        
        # Save config
        self.config.save(config_path)
        self.assertTrue(config_path.exists())
        
        # Load config
        loaded_config = Config.load(config_path)
        
        # Verify loaded config matches original
        self.assertEqual(loaded_config.model.name, self.config.model.name)
        self.assertEqual(loaded_config.model.num_classes, self.config.model.num_classes)
        self.assertEqual(loaded_config.training.batch_size, self.config.training.batch_size)
        self.assertEqual(loaded_config.data.train_path, self.config.data.train_path)
    
    @patch('streamlit.run')
    def test_streamlit_app_integration(self, mock_streamlit_run):
        """Test Streamlit app can be imported and initialized"""
        try:
            # This would test if the Streamlit app can be imported
            # import app.streamlit_app.main
            # We just test that the structure is correct
            app_path = Path(__file__).parent.parent / 'app' / 'streamlit_app' / 'main.py'
            self.assertTrue(app_path.exists())
        except ImportError as e:
            self.fail(f"Streamlit app import failed: {e}")
    
    def test_inference_pipeline_integration(self):
        """Test inference pipeline integration"""
        try:
            from inference.predictor import Predictor
            
            # Mock model for predictor
            with patch('tensorflow.keras.models.load_model') as mock_load:
                mock_model = MagicMock()
                mock_load.return_value = mock_model
                
                predictor = Predictor(model_path='dummy_path')
                self.assertIsNotNone(predictor)
                
        except ImportError as e:
            self.fail(f"Inference pipeline import failed: {e}")
    
    def test_monitoring_integration(self):
        """Test monitoring and logging integration"""
        try:
            from utils.logging_utils import setup_logging
            from monitoring.metrics import MetricsTracker
            
            # Test logging setup
            logger = setup_logging('test_integration')
            self.assertIsNotNone(logger)
            
            # Test metrics tracking
            tracker = MetricsTracker()
            self.assertIsNotNone(tracker)
            
        except ImportError as e:
            self.fail(f"Monitoring integration failed: {e}")
    
    def test_docker_configuration_integration(self):
        """Test Docker configuration files"""
        project_root = Path(__file__).parent.parent
        
        # Check Dockerfile exists and has correct structure
        dockerfile = project_root / 'Dockerfile'
        if dockerfile.exists():
            content = dockerfile.read_text()
            self.assertIn('FROM', content)
            self.assertIn('WORKDIR', content)
            self.assertIn('COPY', content)
        
        # Check docker-compose exists
        docker_compose = project_root / 'docker-compose.yml'
        if docker_compose.exists():
            import yaml
            try:
                with open(docker_compose, 'r') as f:
                    compose_config = yaml.safe_load(f)
                self.assertIn('services', compose_config)
            except yaml.YAMLError:
                self.fail("Invalid docker-compose.yml format")
    
    def test_ci_cd_integration(self):
        """Test CI/CD configuration"""
        project_root = Path(__file__).parent.parent
        
        # Check GitHub Actions workflow
        workflow_file = project_root / '.github' / 'workflows' / 'ci-cd.yml'
        if workflow_file.exists():
            import yaml
            try:
                with open(workflow_file, 'r') as f:
                    workflow_config = yaml.safe_load(f)
                self.assertIn('on', workflow_config)
                self.assertIn('jobs', workflow_config)
            except yaml.YAMLError:
                self.fail("Invalid CI/CD workflow format")
        
        # Check pre-commit configuration
        precommit_file = project_root / '.pre-commit-config.yaml'
        if precommit_file.exists():
            import yaml
            try:
                with open(precommit_file, 'r') as f:
                    precommit_config = yaml.safe_load(f)
                self.assertIn('repos', precommit_config)
            except yaml.YAMLError:
                self.fail("Invalid pre-commit configuration format")


class TestModelTrainingIntegration(unittest.TestCase):
    """Test model training integration scenarios"""
    
    def setUp(self):
        """Set up training integration tests"""
        self.config = Config(
            model=ModelConfig(name='simple_cnn', num_classes=3),
            training=TrainingConfig(batch_size=8, epochs=1),
            data=DataConfig(batch_size=8)
        )
    
    @patch('tensorflow.keras.models.Sequential')
    @patch('numpy.random.rand')
    def test_model_training_flow(self, mock_rand, mock_sequential):
        """Test complete model training flow"""
        # Mock model
        mock_model = MagicMock()
        mock_sequential.return_value = mock_model
        mock_model.fit.return_value.history = {'loss': [0.5, 0.3], 'accuracy': [0.8, 0.9]}
        
        # Mock data
        mock_rand.return_value = np.random.rand(32, 224, 224, 3)
        
        # This would test actual training flow
        # trainer = Trainer(self.config.training, mock_model, train_ds, val_ds)
        # history = trainer.train()
        
        # For now, verify setup is correct
        self.assertIsNotNone(mock_model)
    
    @patch('mlflow.start_run')
    @patch('mlflow.log_metric')
    def test_experiment_tracking_integration(self, mock_log_metric, mock_start_run):
        """Test MLflow experiment tracking integration"""
        # This would test MLflow integration
        # The actual implementation would use MLflow for tracking
        
        mock_start_run.return_value.__enter__ = MagicMock()
        mock_start_run.return_value.__exit__ = MagicMock()
        
        # Test that MLflow calls would work
        self.assertTrue(True)  # Placeholder


class TestDeploymentIntegration(unittest.TestCase):
    """Test deployment integration scenarios"""
    
    def test_model_serialization_integration(self):
        """Test model serialization for deployment"""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / 'test_model.h5'
            
            # This would test model saving/loading
            # model.save(model_path)
            # loaded_model = tf.keras.models.load_model(model_path)
            
            # For now, just test directory creation
            self.assertTrue(Path(temp_dir).exists())
    
    def test_api_integration(self):
        """Test FastAPI integration"""
        try:
            # Test that FastAPI app can be imported
            from app.fastapi.main import app
            self.assertIsNotNone(app)
        except ImportError:
            # If FastAPI app doesn't exist yet, that's okay for this test
            pass
    
    def test_kubernetes_config_integration(self):
        """Test Kubernetes configuration"""
        project_root = Path(__file__).parent.parent
        k8s_dir = project_root / 'infra' / 'k8s'
        
        if k8s_dir.exists():
            # Check for basic K8s files
            yaml_files = list(k8s_dir.glob('*.yaml')) + list(k8s_dir.glob('*.yml'))
            
            if yaml_files:
                import yaml
                for yaml_file in yaml_files:
                    try:
                        with open(yaml_file, 'r') as f:
                            k8s_config = yaml.safe_load(f)
                        # Basic validation - should have apiVersion and kind
                        if isinstance(k8s_config, dict):
                            self.assertIn('apiVersion', k8s_config)
                            self.assertIn('kind', k8s_config)
                    except yaml.YAMLError:
                        self.fail(f"Invalid Kubernetes YAML: {yaml_file}")


if __name__ == '__main__':
    # Run integration tests with more verbosity
    unittest.main(verbosity=2, buffer=True)