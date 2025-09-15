"""
Model Pruning and Quantization for CAPSTONE-LAZARUS
===================================================
"""

import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow import keras
import numpy as np
import logging
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
import json

from src.config import Config

logger = logging.getLogger(__name__)

class ModelPruner:
    """Model pruning utilities using TensorFlow Model Optimization"""
    
    def __init__(self, config: Config):
        self.config = config
        
    def create_pruning_schedule(self, 
                               initial_sparsity: float = 0.0,
                               final_sparsity: float = 0.5,
                               begin_step: int = 0,
                               end_step: int = 1000,
                               frequency: int = 100) -> tfmot.sparsity.keras.PolynomialDecay:
        """
        Create pruning schedule
        
        Args:
            initial_sparsity: Initial pruning ratio
            final_sparsity: Final pruning ratio
            begin_step: Step to begin pruning
            end_step: Step to end pruning
            frequency: Pruning frequency
            
        Returns:
            Pruning schedule
        """
        return tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=initial_sparsity,
            final_sparsity=final_sparsity,
            begin_step=begin_step,
            end_step=end_step,
            power=3,
            frequency=frequency
        )
    
    def apply_magnitude_pruning(self,
                               model: keras.Model,
                               pruning_schedule: Optional[tfmot.sparsity.keras.PruningSchedule] = None,
                               pruning_params: Optional[Dict[str, Any]] = None) -> keras.Model:
        """
        Apply magnitude-based pruning to model
        
        Args:
            model: Model to prune
            pruning_schedule: Custom pruning schedule
            pruning_params: Additional pruning parameters
            
        Returns:
            Pruned model
        """
        logger.info("Applying magnitude-based pruning")
        
        if pruning_schedule is None:
            # Create default schedule
            pruning_schedule = self.create_pruning_schedule()
        
        if pruning_params is None:
            pruning_params = {
                'pruning_schedule': pruning_schedule,
                'block_size': (1, 1),
                'block_pooling_type': 'AVG'
            }
        
        # Apply pruning
        prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
        
        def apply_pruning_to_layer(layer):
            # Only prune Dense and Conv2D layers
            if isinstance(layer, (keras.layers.Dense, keras.layers.Conv2D)):
                return prune_low_magnitude(layer, **pruning_params)
            return layer
        
        pruned_model = tf.keras.models.clone_model(
            model,
            clone_function=apply_pruning_to_layer,
        )
        
        logger.info(f"Pruning applied. Model size: {pruned_model.count_params():,} parameters")
        
        return pruned_model
    
    def apply_structured_pruning(self,
                                model: keras.Model,
                                sparsity_ratio: float = 0.5,
                                block_shape: tuple = (2, 2)) -> keras.Model:
        """
        Apply structured (block) pruning
        
        Args:
            model: Model to prune
            sparsity_ratio: Ratio of weights to prune
            block_shape: Shape of pruning blocks
            
        Returns:
            Pruned model
        """
        logger.info(f"Applying structured pruning with {sparsity_ratio} sparsity")
        
        pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(
                target_sparsity=sparsity_ratio,
                begin_step=0
            ),
            'block_size': block_shape,
            'block_pooling_type': 'AVG'
        }
        
        return self.apply_magnitude_pruning(model, pruning_params=pruning_params)
    
    def strip_pruning(self, model: keras.Model) -> keras.Model:
        """Remove pruning wrappers from model"""
        logger.info("Stripping pruning wrappers")
        
        stripped_model = tfmot.sparsity.keras.strip_pruning(model)
        
        # Verify pruning was applied
        sparsity = self.get_model_sparsity(stripped_model)
        logger.info(f"Final model sparsity: {sparsity:.2%}")
        
        return stripped_model
    
    def get_model_sparsity(self, model: keras.Model) -> float:
        """Calculate overall model sparsity"""
        total_weights = 0
        zero_weights = 0
        
        for layer in model.layers:
            if hasattr(layer, 'kernel'):
                weights = layer.kernel.numpy()
                total_weights += weights.size
                zero_weights += np.count_nonzero(weights == 0)
        
        return zero_weights / total_weights if total_weights > 0 else 0.0
    
    def get_layer_sparsities(self, model: keras.Model) -> Dict[str, float]:
        """Get sparsity for each layer"""
        layer_sparsities = {}
        
        for layer in model.layers:
            if hasattr(layer, 'kernel'):
                weights = layer.kernel.numpy()
                total = weights.size
                zeros = np.count_nonzero(weights == 0)
                layer_sparsities[layer.name] = zeros / total if total > 0 else 0.0
        
        return layer_sparsities

class ModelQuantizer:
    """Model quantization utilities"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def apply_qat(self, 
                  model: keras.Model,
                  quantize_config: Optional[tfmot.quantization.keras.QuantizeConfig] = None) -> keras.Model:
        """
        Apply Quantization Aware Training (QAT)
        
        Args:
            model: Model to quantize
            quantize_config: Custom quantization configuration
            
        Returns:
            QAT model
        """
        logger.info("Applying Quantization Aware Training")
        
        # Use default quantization if no config provided
        if quantize_config is None:
            quantized_model = tfmot.quantization.keras.quantize_model(model)
        else:
            quantized_model = tfmot.quantization.keras.quantize_apply(model, quantize_config)
        
        logger.info(f"QAT applied. Model parameters: {quantized_model.count_params():,}")
        
        return quantized_model
    
    def apply_post_training_quantization(self,
                                       model: keras.Model,
                                       representative_data: Optional[tf.data.Dataset] = None,
                                       optimization: str = 'default') -> str:
        """
        Apply post-training quantization and convert to TFLite
        
        Args:
            model: Trained model
            representative_data: Dataset for calibration
            optimization: Optimization level ('default', 'size', 'latency')
            
        Returns:
            Path to quantized TFLite model
        """
        logger.info("Applying post-training quantization")
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Set optimization flags
        if optimization == 'size':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        elif optimization == 'latency':
            converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
        else:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Representative dataset for full integer quantization
        if representative_data is not None:
            def representative_data_gen():
                for data in representative_data.take(100):
                    # Assume data is (features, labels) tuple
                    if isinstance(data, tuple):
                        yield [tf.cast(data[0], tf.float32)]
                    else:
                        yield [tf.cast(data, tf.float32)]
            
            converter.representative_dataset = representative_data_gen
            
            # Enable full integer quantization
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
        
        # Convert model
        try:
            tflite_model = converter.convert()
            
            # Save quantized model
            model_dir = Path(self.config.models_dir) / "quantized"
            model_dir.mkdir(parents=True, exist_ok=True)
            
            tflite_path = model_dir / f"{model.name}_quantized.tflite"
            
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
            
            logger.info(f"Quantized model saved to {tflite_path}")
            
            # Calculate compression ratio
            original_size = self._get_model_size(model)
            quantized_size = len(tflite_model)
            compression_ratio = original_size / quantized_size
            
            logger.info(f"Compression ratio: {compression_ratio:.2f}x")
            
            return str(tflite_path)
        
        except Exception as e:
            logger.error(f"Quantization failed: {e}")
            raise
    
    def create_int8_quantization(self, 
                                model: keras.Model,
                                calibration_data: tf.data.Dataset) -> str:
        """Create INT8 quantized model"""
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Set INT8 quantization
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        
        # Representative dataset
        def representative_data_gen():
            for data_batch in calibration_data.take(50):
                if isinstance(data_batch, tuple):
                    yield [tf.cast(data_batch[0][:1], tf.float32)]  # Take one sample
                else:
                    yield [tf.cast(data_batch[:1], tf.float32)]
        
        converter.representative_dataset = representative_data_gen
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        
        tflite_model = converter.convert()
        
        # Save model
        model_dir = Path(self.config.models_dir) / "quantized"
        model_dir.mkdir(parents=True, exist_ok=True)
        tflite_path = model_dir / f"{model.name}_int8.tflite"
        
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        logger.info(f"INT8 quantized model saved to {tflite_path}")
        return str(tflite_path)
    
    def benchmark_quantized_model(self, tflite_path: str, test_data: tf.data.Dataset) -> Dict[str, Any]:
        """Benchmark quantized TFLite model"""
        
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        # Get input/output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Run inference on test data
        predictions = []
        inference_times = []
        
        import time
        
        for batch in test_data.take(10):  # Test on 10 batches
            if isinstance(batch, tuple):
                inputs = batch[0].numpy()
            else:
                inputs = batch.numpy()
            
            for i in range(inputs.shape[0]):
                input_data = inputs[i:i+1]
                
                # Set input
                interpreter.set_tensor(input_details[0]['index'], input_data.astype(input_details[0]['dtype']))
                
                # Run inference
                start_time = time.time()
                interpreter.invoke()
                end_time = time.time()
                
                inference_times.append((end_time - start_time) * 1000)  # ms
                
                # Get output
                output_data = interpreter.get_tensor(output_details[0]['index'])
                predictions.append(output_data)
        
        # Calculate metrics
        avg_inference_time = np.mean(inference_times)
        model_size_kb = Path(tflite_path).stat().st_size / 1024
        
        return {
            'model_path': tflite_path,
            'model_size_kb': model_size_kb,
            'avg_inference_time_ms': avg_inference_time,
            'std_inference_time_ms': np.std(inference_times),
            'total_predictions': len(predictions)
        }
    
    def _get_model_size(self, model: keras.Model) -> int:
        """Estimate model size in bytes"""
        total_params = model.count_params()
        return total_params * 4  # Assuming float32 (4 bytes per parameter)

class CompressionPipeline:
    """Complete model compression pipeline"""
    
    def __init__(self, config: Config):
        self.config = config
        self.pruner = ModelPruner(config)
        self.quantizer = ModelQuantizer(config)
    
    def compress_model(self,
                      model: keras.Model,
                      train_data: tf.data.Dataset,
                      val_data: tf.data.Dataset,
                      pruning_ratio: float = 0.5,
                      enable_qat: bool = True,
                      enable_pruning: bool = True) -> Dict[str, Any]:
        """
        Complete model compression pipeline
        
        Args:
            model: Original model
            train_data: Training data
            val_data: Validation data
            pruning_ratio: Target pruning ratio
            enable_qat: Enable quantization aware training
            enable_pruning: Enable pruning
            
        Returns:
            Compression results and model paths
        """
        logger.info("Starting model compression pipeline")
        
        results = {
            'original_model_params': model.count_params(),
            'original_model_size_mb': self.quantizer._get_model_size(model) / (1024 * 1024),
            'compression_stages': []
        }
        
        current_model = model
        
        # Stage 1: Pruning
        if enable_pruning:
            logger.info("Stage 1: Model Pruning")
            
            # Create pruning schedule
            end_step = len(train_data) * 10  # Assume 10 epochs of pruning
            pruning_schedule = self.pruner.create_pruning_schedule(
                final_sparsity=pruning_ratio,
                end_step=end_step
            )
            
            # Apply pruning
            pruned_model = self.pruner.apply_magnitude_pruning(
                current_model,
                pruning_schedule=pruning_schedule
            )
            
            # Compile model
            pruned_model.compile(
                optimizer=current_model.optimizer,
                loss=current_model.loss,
                metrics=current_model.metrics
            )
            
            # Fine-tune pruned model
            pruning_callbacks = [
                tfmot.sparsity.keras.UpdatePruningStep(),
                tfmot.sparsity.keras.PruningSummaries(log_dir=self.config.logs_dir)
            ]
            
            pruned_model.fit(
                train_data,
                validation_data=val_data,
                epochs=5,  # Few epochs for fine-tuning
                callbacks=pruning_callbacks,
                verbose=1
            )
            
            # Strip pruning wrappers
            current_model = self.pruner.strip_pruning(pruned_model)
            
            sparsity = self.pruner.get_model_sparsity(current_model)
            results['compression_stages'].append({
                'stage': 'pruning',
                'sparsity': sparsity,
                'parameters': current_model.count_params()
            })
        
        # Stage 2: Quantization Aware Training
        if enable_qat:
            logger.info("Stage 2: Quantization Aware Training")
            
            qat_model = self.quantizer.apply_qat(current_model)
            
            # Compile QAT model
            qat_model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=1e-5),  # Lower LR for QAT
                loss=current_model.loss,
                metrics=current_model.metrics
            )
            
            # Fine-tune with QAT
            qat_model.fit(
                train_data,
                validation_data=val_data,
                epochs=3,  # Few epochs for QAT fine-tuning
                verbose=1
            )
            
            current_model = qat_model
            
            results['compression_stages'].append({
                'stage': 'quantization_aware_training',
                'parameters': current_model.count_params()
            })
        
        # Stage 3: Post-training quantization (TFLite conversion)
        logger.info("Stage 3: Post-training Quantization")
        
        tflite_path = self.quantizer.apply_post_training_quantization(
            current_model,
            representative_data=train_data.take(50),
            optimization='size'
        )
        
        # Benchmark quantized model
        benchmark_results = self.quantizer.benchmark_quantized_model(
            tflite_path,
            val_data.take(5)
        )
        
        results.update({
            'final_tflite_path': tflite_path,
            'final_model_size_kb': benchmark_results['model_size_kb'],
            'compression_ratio': results['original_model_size_mb'] * 1024 / benchmark_results['model_size_kb'],
            'avg_inference_time_ms': benchmark_results['avg_inference_time_ms'],
            'benchmark_results': benchmark_results
        })
        
        # Save compression report
        self._save_compression_report(results)
        
        logger.info(f"Compression complete. Final size: {results['final_model_size_kb']:.2f} KB")
        logger.info(f"Compression ratio: {results['compression_ratio']:.2f}x")
        
        return results
    
    def _save_compression_report(self, results: Dict[str, Any]):
        """Save compression report"""
        reports_dir = Path(self.config.experiments_dir) / "compression_reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = results.get('timestamp', 'unknown')
        report_path = reports_dir / f"compression_report_{timestamp}.json"
        
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Compression report saved to {report_path}")

# Utility functions
def prune_and_quantize_model(model: keras.Model,
                            config: Config,
                            train_data: tf.data.Dataset,
                            val_data: tf.data.Dataset,
                            pruning_ratio: float = 0.5) -> Dict[str, Any]:
    """Convenience function for model compression"""
    pipeline = CompressionPipeline(config)
    return pipeline.compress_model(
        model, train_data, val_data, 
        pruning_ratio=pruning_ratio
    )