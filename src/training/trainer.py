"""
Training Engine for CAPSTONE-LAZARUS
====================================
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import logging
from typing import Dict, Any, Optional, List, Tuple, Callable
from pathlib import Path
import json
from datetime import datetime
import time

from src.config import Config

logger = logging.getLogger(__name__)

class Trainer:
    """Advanced training engine with comprehensive features"""
    
    def __init__(self, config: Config):
        self.config = config
        self.training_config = config.training
        self.model = None
        self.history = None
        
        # Setup mixed precision if enabled
        if self.training_config.use_mixed_precision:
            policy = keras.mixed_precision.Policy('mixed_float16')
            keras.mixed_precision.set_global_policy(policy)
            logger.info("Mixed precision training enabled")
    
    def train(self, 
              model: keras.Model,
              train_dataset: tf.data.Dataset,
              val_dataset: Optional[tf.data.Dataset] = None,
              callbacks: Optional[List[keras.callbacks.Callback]] = None,
              mlflow_logger: Optional[Any] = None) -> keras.callbacks.History:
        """
        Train model with comprehensive monitoring and callbacks
        
        Args:
            model: Model to train
            train_dataset: Training dataset
            val_dataset: Validation dataset
            callbacks: Additional callbacks
            mlflow_logger: MLflow logger instance
            
        Returns:
            Training history
        """
        self.model = model
        
        logger.info(f"Starting training for {self.training_config.epochs} epochs")
        logger.info(f"Model parameters: {model.count_params():,}")
        
        # Setup callbacks
        training_callbacks = self._setup_callbacks(mlflow_logger)
        if callbacks:
            training_callbacks.extend(callbacks)
        
        # Calculate steps per epoch
        try:
            steps_per_epoch = tf.data.experimental.cardinality(train_dataset).numpy()
            if steps_per_epoch == tf.data.experimental.UNKNOWN_CARDINALITY:
                steps_per_epoch = None
        except:
            steps_per_epoch = None
        
        validation_steps = None
        if val_dataset:
            try:
                validation_steps = tf.data.experimental.cardinality(val_dataset).numpy()
                if validation_steps == tf.data.experimental.UNKNOWN_CARDINALITY:
                    validation_steps = None
            except:
                validation_steps = None
        
        # Start training
        start_time = time.time()
        
        try:
            self.history = model.fit(
                train_dataset,
                epochs=self.training_config.epochs,
                validation_data=val_dataset,
                callbacks=training_callbacks,
                steps_per_epoch=steps_per_epoch,
                validation_steps=validation_steps,
                verbose=1
            )
            
            training_time = time.time() - start_time
            logger.info(f"Training completed in {training_time:.2f} seconds")
            
            # Log final metrics
            if mlflow_logger:
                mlflow_logger.log_metric("training_time_seconds", training_time)
                final_metrics = self._get_final_metrics()
                for metric, value in final_metrics.items():
                    mlflow_logger.log_metric(f"final_{metric}", value)
            
            return self.history
            
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            return self.history
        
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def evaluate(self, 
                model: keras.Model,
                test_dataset: tf.data.Dataset) -> Dict[str, float]:
        """Evaluate model on test dataset"""
        
        logger.info("Evaluating model on test dataset")
        
        # Evaluate
        results = model.evaluate(test_dataset, verbose=1, return_dict=True)
        
        logger.info("Test Results:")
        for metric, value in results.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return results
    
    def predict(self, 
               model: keras.Model,
               dataset: tf.data.Dataset,
               batch_size: Optional[int] = None) -> np.ndarray:
        """Generate predictions"""
        
        logger.info("Generating predictions")
        
        predictions = model.predict(
            dataset,
            batch_size=batch_size,
            verbose=1
        )
        
        logger.info(f"Generated {len(predictions)} predictions")
        
        return predictions
    
    def _setup_callbacks(self, mlflow_logger: Optional[Any] = None) -> List[keras.callbacks.Callback]:
        """Setup training callbacks"""
        
        callbacks = []
        
        # Model checkpointing
        checkpoint_dir = Path(self.config.models_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_dir / "model_epoch_{epoch:02d}_val_acc_{val_accuracy:.4f}.h5"),
            monitor='val_accuracy' if 'accuracy' in self.training_config.metrics else 'val_loss',
            save_best_only=self.training_config.save_best_only,
            save_weights_only=False,
            mode='max' if 'accuracy' in str(self.training_config.metrics) else 'min',
            verbose=1
        )
        callbacks.append(checkpoint_callback)
        
        # Early stopping
        if self.training_config.early_stopping_patience > 0:
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_accuracy' if 'accuracy' in self.training_config.metrics else 'val_loss',
                patience=self.training_config.early_stopping_patience,
                restore_best_weights=True,
                mode='max' if 'accuracy' in str(self.training_config.metrics) else 'min',
                verbose=1
            )
            callbacks.append(early_stopping)
        
        # Reduce learning rate on plateau
        if self.training_config.reduce_lr_patience > 0:
            reduce_lr = keras.callbacks.ReduceLROnPlateau(
                monitor='val_accuracy' if 'accuracy' in self.training_config.metrics else 'val_loss',
                factor=0.5,
                patience=self.training_config.reduce_lr_patience,
                mode='max' if 'accuracy' in str(self.training_config.metrics) else 'min',
                verbose=1,
                min_lr=1e-7
            )
            callbacks.append(reduce_lr)
        
        # TensorBoard logging
        tensorboard_dir = Path(self.config.logs_dir) / "tensorboard"
        tensorboard_dir.mkdir(parents=True, exist_ok=True)
        
        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir=str(tensorboard_dir),
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            update_freq='epoch'
        )
        callbacks.append(tensorboard_callback)
        
        # Custom metrics callback
        if mlflow_logger:
            mlflow_callback = MLflowCallback(mlflow_logger)
            callbacks.append(mlflow_callback)
        
        # Learning rate scheduler (optional)
        if hasattr(self.training_config, 'lr_schedule') and self.training_config.lr_schedule:
            lr_callback = self._create_lr_scheduler()
            if lr_callback:
                callbacks.append(lr_callback)
        
        # Progress callback
        progress_callback = TrainingProgressCallback()
        callbacks.append(progress_callback)
        
        logger.info(f"Setup {len(callbacks)} training callbacks")
        
        return callbacks
    
    def _create_lr_scheduler(self) -> Optional[keras.callbacks.Callback]:
        """Create learning rate scheduler"""
        
        def cosine_decay_with_warmup(epoch, lr):
            if epoch < 5:  # Warmup
                return lr * (epoch + 1) / 5
            else:  # Cosine decay
                progress = (epoch - 5) / (self.training_config.epochs - 5)
                return lr * 0.5 * (1 + np.cos(np.pi * progress))
        
        return keras.callbacks.LearningRateScheduler(cosine_decay_with_warmup, verbose=1)
    
    def _get_final_metrics(self) -> Dict[str, float]:
        """Get final training metrics"""
        if not self.history:
            return {}
        
        final_metrics = {}
        
        for metric in self.history.history.keys():
            values = self.history.history[metric]
            if values:
                final_metrics[metric] = values[-1]
        
        return final_metrics
    
    def save_training_history(self, output_path: str):
        """Save training history to file"""
        if not self.history:
            logger.warning("No training history to save")
            return
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert history to serializable format
        history_data = {
            'history': self.history.history,
            'epochs': len(self.history.history.get('loss', [])),
            'params': self.history.params,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(history_data, f, indent=2, default=str)
        
        logger.info(f"Training history saved to {output_path}")
    
    def load_training_history(self, input_path: str) -> bool:
        """Load training history from file"""
        input_path = Path(input_path)
        
        if not input_path.exists():
            logger.error(f"History file not found: {input_path}")
            return False
        
        try:
            with open(input_path, 'r') as f:
                history_data = json.load(f)
            
            # Reconstruct history object (simplified)
            class MockHistory:
                def __init__(self, history_dict, params):
                    self.history = history_dict
                    self.params = params
            
            self.history = MockHistory(
                history_data['history'],
                history_data.get('params', {})
            )
            
            logger.info(f"Training history loaded from {input_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load training history: {e}")
            return False


class MLflowCallback(keras.callbacks.Callback):
    """Custom callback for MLflow logging"""
    
    def __init__(self, mlflow_logger):
        super().__init__()
        self.mlflow_logger = mlflow_logger
    
    def on_epoch_end(self, epoch, logs=None):
        if logs and self.mlflow_logger:
            for metric, value in logs.items():
                self.mlflow_logger.log_metric(metric, value, step=epoch)


class TrainingProgressCallback(keras.callbacks.Callback):
    """Custom callback for training progress monitoring"""
    
    def __init__(self):
        super().__init__()
        self.epoch_start_time = None
        self.training_start_time = None
    
    def on_train_begin(self, logs=None):
        self.training_start_time = time.time()
        logger.info("Training started")
    
    def on_train_end(self, logs=None):
        if self.training_start_time:
            total_time = time.time() - self.training_start_time
            logger.info(f"Training completed in {total_time:.2f} seconds")
    
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
    
    def on_epoch_end(self, epoch, logs=None):
        if self.epoch_start_time:
            epoch_time = time.time() - self.epoch_start_time
            
            log_msg = f"Epoch {epoch + 1} completed in {epoch_time:.2f}s"
            if logs:
                key_metrics = ['loss', 'accuracy', 'val_loss', 'val_accuracy']
                metrics_str = []
                for metric in key_metrics:
                    if metric in logs:
                        metrics_str.append(f"{metric}: {logs[metric]:.4f}")
                
                if metrics_str:
                    log_msg += f" - {' - '.join(metrics_str)}"
            
            logger.info(log_msg)


class GradualUnfreezing:
    """Implement gradual unfreezing for transfer learning"""
    
    def __init__(self, model: keras.Model, unfreeze_schedule: List[int]):
        """
        Args:
            model: Model with frozen layers
            unfreeze_schedule: Epochs at which to unfreeze layers
        """
        self.model = model
        self.unfreeze_schedule = sorted(unfreeze_schedule)
        self.frozen_layers = [layer for layer in model.layers if not layer.trainable]
    
    def create_callback(self) -> keras.callbacks.Callback:
        """Create callback for gradual unfreezing"""
        
        class UnfreezingCallback(keras.callbacks.Callback):
            def __init__(self, unfreezer):
                super().__init__()
                self.unfreezer = unfreezer
            
            def on_epoch_begin(self, epoch, logs=None):
                if epoch in self.unfreezer.unfreeze_schedule:
                    self.unfreezer.unfreeze_next_layers()
        
        return UnfreezingCallback(self)
    
    def unfreeze_next_layers(self):
        """Unfreeze next batch of layers"""
        if not self.frozen_layers:
            return
        
        # Unfreeze 10% of remaining frozen layers
        num_to_unfreeze = max(1, len(self.frozen_layers) // 10)
        
        for _ in range(num_to_unfreeze):
            if self.frozen_layers:
                layer = self.frozen_layers.pop()
                layer.trainable = True
                logger.info(f"Unfroze layer: {layer.name}")


class CurriculumLearning:
    """Implement curriculum learning strategies"""
    
    def __init__(self, easy_data: tf.data.Dataset, hard_data: tf.data.Dataset):
        self.easy_data = easy_data
        self.hard_data = hard_data
    
    def create_curriculum_dataset(self, 
                                 epoch: int, 
                                 total_epochs: int,
                                 strategy: str = "linear") -> tf.data.Dataset:
        """Create curriculum dataset based on epoch"""
        
        if strategy == "linear":
            # Linear increase in hard examples
            hard_ratio = epoch / total_epochs
        elif strategy == "exponential":
            # Exponential increase in hard examples
            hard_ratio = (np.exp(epoch / total_epochs * 3) - 1) / (np.exp(3) - 1)
        else:
            hard_ratio = 0.5  # Default 50/50
        
        easy_ratio = 1 - hard_ratio
        
        # Sample from datasets
        easy_size = int(1000 * easy_ratio)  # Assuming 1000 samples total
        hard_size = int(1000 * hard_ratio)
        
        easy_samples = self.easy_data.take(easy_size)
        hard_samples = self.hard_data.take(hard_size)
        
        # Combine datasets
        curriculum_dataset = easy_samples.concatenate(hard_samples)
        curriculum_dataset = curriculum_dataset.shuffle(1000)
        
        return curriculum_dataset


# Utility functions
def create_trainer(config: Config) -> Trainer:
    """Factory function for creating trainer"""
    return Trainer(config)


def train_with_mixed_precision(model: keras.Model,
                              train_dataset: tf.data.Dataset,
                              val_dataset: tf.data.Dataset,
                              config: Config) -> keras.callbacks.History:
    """Train model with automatic mixed precision"""
    
    # Enable mixed precision
    policy = keras.mixed_precision.Policy('mixed_float16')
    keras.mixed_precision.set_global_policy(policy)
    
    # Ensure loss scaling for mixed precision
    if hasattr(model.optimizer, 'loss_scale'):
        model.optimizer = keras.mixed_precision.LossScaleOptimizer(model.optimizer)
    
    trainer = Trainer(config)
    return trainer.train(model, train_dataset, val_dataset)