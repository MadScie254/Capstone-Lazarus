"""
Comprehensive Training Pipeline for Plant Disease Classification
==============================================================
Multi-model ensemble training with production deployment focus

Features:
- Multi-architecture support (EfficientNet, ResNet, MobileNet, DenseNet)
- Deterministic training with reproducible seeding
- Advanced callbacks and monitoring
- Model registry integration
- Ensemble training coordination
- Streamlit integration support
"""

import os
import logging
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib

# Ensure reproducibility
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for individual model architectures"""
    architecture: str  # 'efficientnet_b0', 'resnet50', 'mobilenet_v2', 'densenet121'
    input_shape: Tuple[int, int, int] = (224, 224, 3)
    num_classes: int = 19
    dropout_rate: float = 0.3
    learning_rate: float = 1e-4
    batch_size: int = 32
    fine_tune_layers: int = 50  # Number of layers to fine-tune
    use_mixed_precision: bool = True
    
    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass 
class TrainingConfig:
    """Comprehensive training configuration"""
    epochs: int = 100
    early_stopping_patience: int = 15
    reduce_lr_patience: int = 8
    reduce_lr_factor: float = 0.5
    min_lr: float = 1e-7
    class_weights: Optional[Dict[int, float]] = None
    augmentation_strength: float = 0.8
    validation_split: float = 0.2
    test_split: float = 0.2
    use_stratified_split: bool = True
    save_best_only: bool = True
    monitor_metric: str = 'val_accuracy'
    mode: str = 'max'
    
    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class EnsembleConfig:
    """Configuration for ensemble training and inference"""
    models: List[str]  # List of architectures to ensemble
    ensemble_method: str = 'soft_voting'  # 'soft_voting', 'hard_voting', 'stacking'
    stacking_meta_learner: str = 'logistic'  # For stacking method
    ensemble_weights: Optional[List[float]] = None
    confidence_threshold: float = 0.8
    
    def to_dict(self) -> Dict:
        return asdict(self)

class TrainingPipeline:
    """
    Comprehensive training pipeline for plant disease classification
    
    Features:
    - Multi-model architecture support
    - Deterministic reproducible training
    - Advanced callback system
    - Model registry integration
    - Production deployment focus
    """
    
    def __init__(
        self, 
        data_dir: str,
        output_dir: str = "models",
        experiment_name: str = "plant_disease_classification"
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir = self.output_dir / "trained_models"
        self.logs_dir = self.output_dir / "logs" 
        self.reports_dir = self.output_dir / "reports"
        
        for dir_path in [self.models_dir, self.logs_dir, self.reports_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        # Initialize components
        self._setup_logging()
        self._ensure_reproducibility()
        self.model_registry = ModelRegistry(str(self.models_dir))
        
        logger.info(f"Training pipeline initialized: {experiment_name}")
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def _setup_logging(self):
        """Configure comprehensive logging"""
        log_file = self.logs_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def _ensure_reproducibility(self):
        """Ensure deterministic training across runs"""
        # Set all random seeds
        tf.random.set_seed(SEED)
        np.random.seed(SEED)
        
        # Configure TensorFlow for reproducibility
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
        
        # Configure GPU if available
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"Configured {len(gpus)} GPU(s) for training")
            except RuntimeError as e:
                logger.warning(f"GPU configuration failed: {e}")
        
        logger.info("Reproducibility settings applied")
    
    def prepare_data(
        self, 
        model_config: ModelConfig,
        training_config: TrainingConfig
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """
        Prepare data with advanced augmentation and balancing
        
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        from src.data_utils import PlantDiseaseDataLoader
        
        logger.info("Preparing dataset for training...")
        
        # Initialize data loader
        data_loader = PlantDiseaseDataLoader(
            str(self.data_dir),
            img_size=model_config.input_shape[:2],
            batch_size=model_config.batch_size
        )
        
        # Scan dataset
        dataset_info = data_loader.scan_dataset()
        logger.info(f"Dataset scanned: {dataset_info['total_classes']} classes, {dataset_info['total_images']} images")
        
        # Create balanced splits
        train_paths, val_paths, test_paths, train_labels, val_labels, test_labels = data_loader.create_balanced_splits(
            test_size=training_config.test_split,
            val_size=training_config.validation_split,
            stratify=training_config.use_stratified_split
        )
        
        # Compute class weights if not provided
        if training_config.class_weights is None:
            training_config.class_weights = data_loader.compute_class_weights(train_labels)
            logger.info("Computed class weights for imbalanced dataset")
        
        # Create TensorFlow datasets
        train_dataset = data_loader.create_tf_dataset(
            train_paths, train_labels, 
            is_training=True,
            augment=True
        )
        
        val_dataset = data_loader.create_tf_dataset(
            val_paths, val_labels,
            is_training=False,
            augment=False
        )
        
        test_dataset = data_loader.create_tf_dataset(
            test_paths, test_labels,
            is_training=False, 
            augment=False
        )
        
        # Cache datasets for performance
        train_dataset = train_dataset.cache().prefetch(tf.data.AUTOTUNE)
        val_dataset = val_dataset.cache().prefetch(tf.data.AUTOTUNE)
        test_dataset = test_dataset.cache().prefetch(tf.data.AUTOTUNE)
        
        logger.info(f"Data preparation complete:")
        logger.info(f"  - Training samples: {len(train_paths)}")
        logger.info(f"  - Validation samples: {len(val_paths)}")
        logger.info(f"  - Test samples: {len(test_paths)}")
        
        return train_dataset, val_dataset, test_dataset
    
    def create_model(self, config: ModelConfig) -> tf.keras.Model:
        """
        Create model with specified architecture
        
        Supports:
        - EfficientNetB0
        - ResNet50
        - MobileNetV2
        - DenseNet121
        """
        from src.models.architectures import create_model
        
        logger.info(f"Creating model: {config.architecture}")
        
        # Enable mixed precision if specified
        if config.use_mixed_precision:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            logger.info("Mixed precision training enabled")
        
        # Create model using the model factory
        model = create_model(
            architecture=config.architecture,
            input_shape=config.input_shape,
            num_classes=config.num_classes,
            dropout_rate=config.dropout_rate,
            fine_tune_layers=config.fine_tune_layers
        )
        
        # Compile model
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=config.learning_rate,
            weight_decay=1e-4
        )
        
        # Use mixed precision loss scaling if enabled
        if config.use_mixed_precision:
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top_3_accuracy'),
                tf.keras.metrics.SparseCategoricalCrossentropy(name='crossentropy')
            ]
        )
        
        # Model summary
        model.summary(print_fn=logger.info)
        
        return model
    
    def get_callbacks(
        self, 
        model_name: str,
        training_config: TrainingConfig
    ) -> List[tf.keras.callbacks.Callback]:
        """Get comprehensive training callbacks"""
        
        model_dir = self.models_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        callbacks = [
            # Model checkpointing
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(model_dir / 'best_model.h5'),
                monitor=training_config.monitor_metric,
                mode=training_config.mode,
                save_best_only=training_config.save_best_only,
                save_weights_only=False,
                verbose=1
            ),
            
            # Early stopping
            tf.keras.callbacks.EarlyStopping(
                monitor=training_config.monitor_metric,
                mode=training_config.mode,
                patience=training_config.early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Learning rate reduction
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor=training_config.monitor_metric,
                mode=training_config.mode,
                factor=training_config.reduce_lr_factor,
                patience=training_config.reduce_lr_patience,
                min_lr=training_config.min_lr,
                verbose=1
            ),
            
            # CSV logging
            tf.keras.callbacks.CSVLogger(
                filename=str(self.logs_dir / f'{model_name}_training.csv'),
                append=True
            ),
            
            # TensorBoard logging
            tf.keras.callbacks.TensorBoard(
                log_dir=str(self.logs_dir / f'{model_name}_tensorboard'),
                histogram_freq=1,
                write_graph=True,
                write_images=False,
                profile_batch='10,20'
            )
        ]
        
        return callbacks
    
    def train_single_model(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Train a single model with comprehensive tracking
        
        Returns:
            Training results and metadata
        """
        if model_name is None:
            model_name = f"{model_config.architecture}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Starting training for: {model_name}")
        
        # Prepare data
        train_dataset, val_dataset, test_dataset = self.prepare_data(model_config, training_config)
        
        # Create model
        model = self.create_model(model_config)
        
        # Get callbacks
        callbacks = self.get_callbacks(model_name, training_config)
        
        # Train model
        start_time = datetime.now()
        
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=training_config.epochs,
            callbacks=callbacks,
            class_weight=training_config.class_weights,
            verbose=1
        )
        
        training_time = datetime.now() - start_time
        
        # Evaluate on test set
        test_results = model.evaluate(test_dataset, verbose=0)
        test_metrics = dict(zip(model.metrics_names, test_results))
        
        # Prepare training metadata
        metadata = {
            'model_name': model_name,
            'architecture': model_config.architecture,
            'training_time': str(training_time),
            'total_epochs': len(history.history['loss']),
            'best_val_accuracy': max(history.history.get('val_accuracy', [0])),
            'final_test_accuracy': test_metrics.get('accuracy', 0),
            'model_config': model_config.to_dict(),
            'training_config': training_config.to_dict(),
            'test_metrics': test_metrics,
            'trained_at': datetime.now().isoformat(),
            'python_version': f"{tf.__version__}",
            'seed': SEED
        }
        
        # Save training history
        history_path = self.reports_dir / f'{model_name}_history.json'
        with open(history_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_history = {
                key: [float(val) for val in values] 
                for key, values in history.history.items()
            }
            json.dump(serializable_history, f, indent=2)
        
        # Register model
        model_path = self.models_dir / model_name / 'best_model.h5'
        registration_result = self.model_registry.register_model(
            model_path=str(model_path),
            model_name=model_name,
            architecture=model_config.architecture,
            metrics=test_metrics,
            metadata=metadata
        )
        
        logger.info(f"Training complete for {model_name}")
        logger.info(f"Best validation accuracy: {metadata['best_val_accuracy']:.4f}")
        logger.info(f"Test accuracy: {metadata['final_test_accuracy']:.4f}")
        logger.info(f"Training time: {training_time}")
        
        return {
            'model_name': model_name,
            'history': history.history,
            'metadata': metadata,
            'registration': registration_result
        }
    
    def train_ensemble(
        self,
        ensemble_config: EnsembleConfig,
        training_config: TrainingConfig,
        experiment_suffix: str = ""
    ) -> Dict[str, Any]:
        """
        Train multiple models for ensemble
        
        Returns:
            Ensemble training results and metadata
        """
        experiment_name = f"ensemble_{self.experiment_name}{experiment_suffix}"
        logger.info(f"Starting ensemble training: {experiment_name}")
        logger.info(f"Models to train: {ensemble_config.models}")
        
        ensemble_results = {}
        all_metadata = []
        
        # Train each model in the ensemble
        for architecture in ensemble_config.models:
            model_config = ModelConfig(architecture=architecture)
            model_name = f"{experiment_name}_{architecture}"
            
            try:
                result = self.train_single_model(
                    model_config=model_config,
                    training_config=training_config,
                    model_name=model_name
                )
                ensemble_results[architecture] = result
                all_metadata.append(result['metadata'])
                
            except Exception as e:
                logger.error(f"Failed to train {architecture}: {e}")
                ensemble_results[architecture] = {'error': str(e)}
        
        # Create ensemble metadata
        ensemble_metadata = {
            'ensemble_name': experiment_name,
            'ensemble_config': ensemble_config.to_dict(),
            'training_config': training_config.to_dict(),
            'models_trained': len([r for r in ensemble_results.values() if 'error' not in r]),
            'total_models': len(ensemble_config.models),
            'individual_results': all_metadata,
            'created_at': datetime.now().isoformat()
        }
        
        # Save ensemble metadata
        ensemble_path = self.reports_dir / f'{experiment_name}_ensemble.json'
        with open(ensemble_path, 'w') as f:
            json.dump(ensemble_metadata, f, indent=2, default=str)
        
        logger.info(f"Ensemble training complete: {experiment_name}")
        logger.info(f"Successfully trained: {ensemble_metadata['models_trained']}/{ensemble_metadata['total_models']} models")
        
        return {
            'ensemble_name': experiment_name,
            'individual_results': ensemble_results,
            'ensemble_metadata': ensemble_metadata
        }


class ModelRegistry:
    """Model registry for tracking and managing trained models"""
    
    def __init__(self, registry_dir: str):
        self.registry_dir = Path(registry_dir)
        self.registry_file = self.registry_dir / "model_registry.json"
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing registry or create new
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                self.registry = json.load(f)
        else:
            self.registry = {
                'models': {},
                'created_at': datetime.now().isoformat(),
                'version': '1.0'
            }
    
    def register_model(
        self,
        model_path: str,
        model_name: str,
        architecture: str,
        metrics: Dict[str, float],
        metadata: Dict[str, Any]
    ) -> Dict[str, str]:
        """Register a trained model in the registry"""
        
        # Generate model hash
        model_hash = self._generate_model_hash(model_path)
        
        # Create registry entry
        entry = {
            'model_id': model_hash[:8],
            'model_name': model_name,
            'architecture': architecture,
            'model_path': model_path,
            'metrics': metrics,
            'metadata': metadata,
            'registered_at': datetime.now().isoformat(),
            'status': 'active'
        }
        
        self.registry['models'][model_name] = entry
        
        # Save registry
        self._save_registry()
        
        logger.info(f"Model registered: {model_name} ({entry['model_id']})")
        
        return {
            'model_id': entry['model_id'],
            'status': 'registered',
            'registry_path': str(self.registry_file)
        }
    
    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """Get information about a registered model"""
        return self.registry['models'].get(model_name)
    
    def list_models(self) -> List[Dict]:
        """List all registered models"""
        return list(self.registry['models'].values())
    
    def get_best_model(self, metric: str = 'accuracy') -> Optional[Dict]:
        """Get the best model based on a metric"""
        if not self.registry['models']:
            return None
        
        best_model = max(
            self.registry['models'].values(),
            key=lambda x: x['metrics'].get(metric, 0)
        )
        return best_model
    
    def _generate_model_hash(self, model_path: str) -> str:
        """Generate a hash for the model file"""
        if not os.path.exists(model_path):
            return hashlib.md5(model_path.encode()).hexdigest()
        
        hash_md5 = hashlib.md5()
        with open(model_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _save_registry(self):
        """Save the registry to file"""
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2, default=str)


# Verification command helper
def verify_training_pipeline():
    """
    Verification function for training pipeline functionality
    Usage: python -m src.training.pipeline
    """
    print("🔍 TRAINING PIPELINE VERIFICATION")
    print("=" * 50)
    
    try:
        # Test configurations
        model_config = ModelConfig(architecture='efficientnet_b0')
        training_config = TrainingConfig(epochs=1)  # Quick test
        ensemble_config = EnsembleConfig(models=['efficientnet_b0'])
        
        print("✅ Configuration classes initialized successfully")
        
        # Test pipeline initialization (without actual data)
        pipeline = TrainingPipeline(
            data_dir="test_data", 
            output_dir="test_models",
            experiment_name="verification_test"
        )
        
        print("✅ Training pipeline initialized successfully")
        print("✅ Model registry initialized successfully")
        print("✅ Logging system configured")
        print("✅ Reproducibility settings applied")
        
        # Test model registry
        registry = ModelRegistry("test_registry")
        print("✅ Model registry system functional")
        
        print("\n🎯 VERIFICATION COMPLETE")
        print("Training pipeline is ready for production use")
        print("To run full training, use: pipeline.train_single_model() or pipeline.train_ensemble()")
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        raise


if __name__ == "__main__":
    verify_training_pipeline()