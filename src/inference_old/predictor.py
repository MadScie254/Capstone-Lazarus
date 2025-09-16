"""
Prediction Engine for CAPSTONE-LAZARUS
=====================================
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
import json
import time
from concurrent.futures import ThreadPoolExecutor
import asyncio
from dataclasses import dataclass
from datetime import datetime

from src.config import Config

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Structure for prediction results"""
    predictions: np.ndarray
    probabilities: np.ndarray
    confidence: float
    predicted_class: str
    predicted_index: int
    processing_time: float
    metadata: Dict[str, Any]


class Predictor:
    """Advanced prediction engine with optimization and serving capabilities"""
    
    def __init__(self, config: Config):
        self.config = config
        self.models = {}
        self.class_labels = self._load_class_labels()
        self.preprocessing_fn = None
        
    def load_model(self, 
                   model_path: str, 
                   model_name: str = "default",
                   compile_model: bool = True) -> bool:
        """
        Load a trained model for inference
        
        Args:
            model_path: Path to saved model
            model_name: Name to store model under
            compile_model: Whether to compile the model
            
        Returns:
            Success status
        """
        try:
            model_path = Path(model_path)
            
            if not model_path.exists():
                logger.error(f"Model not found: {model_path}")
                return False
            
            logger.info(f"Loading model from {model_path}")
            
            # Load model based on format
            if model_path.suffix == '.h5':
                model = keras.models.load_model(str(model_path), compile=compile_model)
            elif model_path.is_dir():
                model = keras.models.load_model(str(model_path), compile=compile_model)
            else:
                logger.error(f"Unsupported model format: {model_path.suffix}")
                return False
            
            # Optimize for inference
            if hasattr(model, 'predict'):
                # Warm up the model
                dummy_input = np.random.random((1,) + model.input_shape[1:]).astype(np.float32)
                _ = model.predict(dummy_input, verbose=0)
                logger.info("Model warmed up successfully")
            
            self.models[model_name] = model
            logger.info(f"Model '{model_name}' loaded successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def predict_single(self, 
                      image: Union[np.ndarray, tf.Tensor], 
                      model_name: str = "default",
                      return_probabilities: bool = True) -> PredictionResult:
        """
        Make prediction on single image
        
        Args:
            image: Input image
            model_name: Name of model to use
            return_probabilities: Whether to return class probabilities
            
        Returns:
            Prediction result
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not loaded")
        
        model = self.models[model_name]
        start_time = time.time()
        
        try:
            # Preprocess image
            processed_image = self._preprocess_image(image, model.input_shape[1:3])
            
            # Add batch dimension if needed
            if len(processed_image.shape) == 3:
                processed_image = tf.expand_dims(processed_image, 0)
            
            # Make prediction
            predictions = model(processed_image, training=False)
            
            if return_probabilities:
                probabilities = tf.nn.softmax(predictions).numpy()[0]
            else:
                probabilities = predictions.numpy()[0]
            
            # Get predicted class
            predicted_index = int(np.argmax(probabilities))
            confidence = float(probabilities[predicted_index])
            
            predicted_class = self.class_labels.get(predicted_index, f"Class_{predicted_index}")
            
            processing_time = time.time() - start_time
            
            return PredictionResult(
                predictions=predictions.numpy()[0],
                probabilities=probabilities,
                confidence=confidence,
                predicted_class=predicted_class,
                predicted_index=predicted_index,
                processing_time=processing_time,
                metadata={
                    'model_name': model_name,
                    'timestamp': datetime.now().isoformat(),
                    'input_shape': processed_image.shape
                }
            )
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise
    
    def predict_batch(self, 
                     images: List[Union[np.ndarray, tf.Tensor]], 
                     model_name: str = "default",
                     batch_size: int = 32,
                     return_probabilities: bool = True) -> List[PredictionResult]:
        """
        Make predictions on batch of images
        
        Args:
            images: List of input images
            model_name: Name of model to use
            batch_size: Batch size for processing
            return_probabilities: Whether to return class probabilities
            
        Returns:
            List of prediction results
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not loaded")
        
        model = self.models[model_name]
        results = []
        
        logger.info(f"Processing {len(images)} images in batches of {batch_size}")
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            start_time = time.time()
            
            try:
                # Preprocess batch
                processed_batch = []
                for img in batch_images:
                    processed_img = self._preprocess_image(img, model.input_shape[1:3])
                    processed_batch.append(processed_img)
                
                # Stack into batch
                batch_tensor = tf.stack(processed_batch)
                
                # Make predictions
                batch_predictions = model(batch_tensor, training=False)
                
                if return_probabilities:
                    batch_probabilities = tf.nn.softmax(batch_predictions).numpy()
                else:
                    batch_probabilities = batch_predictions.numpy()
                
                processing_time = time.time() - start_time
                
                # Process each result
                for j, (pred, prob) in enumerate(zip(batch_predictions.numpy(), batch_probabilities)):
                    predicted_index = int(np.argmax(prob))
                    confidence = float(prob[predicted_index])
                    predicted_class = self.class_labels.get(predicted_index, f"Class_{predicted_index}")
                    
                    result = PredictionResult(
                        predictions=pred,
                        probabilities=prob,
                        confidence=confidence,
                        predicted_class=predicted_class,
                        predicted_index=predicted_index,
                        processing_time=processing_time / len(batch_images),
                        metadata={
                            'model_name': model_name,
                            'timestamp': datetime.now().isoformat(),
                            'batch_index': i + j,
                            'batch_size': len(batch_images)
                        }
                    )
                    
                    results.append(result)
                
                logger.info(f"Processed batch {i//batch_size + 1}/{(len(images) + batch_size - 1)//batch_size}")
                
            except Exception as e:
                logger.error(f"Batch prediction error: {e}")
                # Add error results for this batch
                for j in range(len(batch_images)):
                    error_result = PredictionResult(
                        predictions=np.array([]),
                        probabilities=np.array([]),
                        confidence=0.0,
                        predicted_class="Error",
                        predicted_index=-1,
                        processing_time=0.0,
                        metadata={'error': str(e), 'batch_index': i + j}
                    )
                    results.append(error_result)
        
        return results
    
    async def predict_async(self, 
                           images: List[Union[np.ndarray, tf.Tensor]], 
                           model_name: str = "default",
                           max_workers: int = 4) -> List[PredictionResult]:
        """
        Asynchronous batch prediction
        
        Args:
            images: List of input images
            model_name: Name of model to use
            max_workers: Maximum number of worker threads
            
        Returns:
            List of prediction results
        """
        loop = asyncio.get_event_loop()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            tasks = []
            
            for image in images:
                task = loop.run_in_executor(
                    executor, 
                    self.predict_single, 
                    image, 
                    model_name
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Async prediction error for image {i}: {result}")
                error_result = PredictionResult(
                    predictions=np.array([]),
                    probabilities=np.array([]),
                    confidence=0.0,
                    predicted_class="Error",
                    predicted_index=-1,
                    processing_time=0.0,
                    metadata={'error': str(result), 'image_index': i}
                )
                processed_results.append(error_result)
            else:
                processed_results.append(result)
        
        return processed_results
    
    def predict_with_ensemble(self, 
                             image: Union[np.ndarray, tf.Tensor],
                             model_names: List[str],
                             voting: str = "soft") -> PredictionResult:
        """
        Make ensemble prediction using multiple models
        
        Args:
            image: Input image
            model_names: List of model names to use
            voting: Voting strategy ('soft' or 'hard')
            
        Returns:
            Ensemble prediction result
        """
        if not model_names:
            raise ValueError("No models specified for ensemble")
        
        for model_name in model_names:
            if model_name not in self.models:
                raise ValueError(f"Model '{model_name}' not loaded")
        
        start_time = time.time()
        
        # Get predictions from all models
        individual_results = []
        for model_name in model_names:
            result = self.predict_single(image, model_name)
            individual_results.append(result)
        
        if voting == "soft":
            # Average probabilities
            all_probabilities = np.stack([r.probabilities for r in individual_results])
            ensemble_probabilities = np.mean(all_probabilities, axis=0)
            
        elif voting == "hard":
            # Majority voting
            predicted_classes = [r.predicted_index for r in individual_results]
            ensemble_probabilities = np.zeros(len(self.class_labels))
            
            for class_idx in predicted_classes:
                ensemble_probabilities[class_idx] += 1
            
            ensemble_probabilities /= len(model_names)
        
        else:
            raise ValueError(f"Unknown voting strategy: {voting}")
        
        # Get final prediction
        predicted_index = int(np.argmax(ensemble_probabilities))
        confidence = float(ensemble_probabilities[predicted_index])
        predicted_class = self.class_labels.get(predicted_index, f"Class_{predicted_index}")
        
        processing_time = time.time() - start_time
        
        return PredictionResult(
            predictions=ensemble_probabilities,
            probabilities=ensemble_probabilities,
            confidence=confidence,
            predicted_class=predicted_class,
            predicted_index=predicted_index,
            processing_time=processing_time,
            metadata={
                'ensemble_models': model_names,
                'voting_strategy': voting,
                'timestamp': datetime.now().isoformat(),
                'individual_results': [
                    {
                        'model': r.metadata['model_name'],
                        'prediction': r.predicted_class,
                        'confidence': r.confidence
                    }
                    for r in individual_results
                ]
            }
        )
    
    def benchmark_model(self, 
                       model_name: str,
                       num_samples: int = 100,
                       warmup_runs: int = 10) -> Dict[str, float]:
        """
        Benchmark model performance
        
        Args:
            model_name: Name of model to benchmark
            num_samples: Number of samples for benchmarking
            warmup_runs: Number of warmup runs
            
        Returns:
            Benchmark results
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not loaded")
        
        model = self.models[model_name]
        input_shape = model.input_shape[1:]
        
        logger.info(f"Benchmarking model '{model_name}' with {num_samples} samples")
        
        # Generate random test data
        test_data = np.random.random((num_samples,) + input_shape).astype(np.float32)
        
        # Warmup
        warmup_data = test_data[:warmup_runs]
        for _ in range(warmup_runs):
            _ = model.predict(warmup_data, verbose=0)
        
        # Benchmark single predictions
        single_times = []
        for i in range(min(50, num_samples)):  # Test up to 50 single predictions
            start_time = time.time()
            _ = model.predict(test_data[i:i+1], verbose=0)
            single_times.append(time.time() - start_time)
        
        # Benchmark batch predictions
        batch_sizes = [1, 8, 16, 32, 64]
        batch_results = {}
        
        for batch_size in batch_sizes:
            if batch_size <= num_samples:
                batch_data = test_data[:batch_size]
                
                start_time = time.time()
                _ = model.predict(batch_data, verbose=0)
                batch_time = time.time() - start_time
                
                batch_results[f'batch_{batch_size}'] = {
                    'total_time': batch_time,
                    'per_sample': batch_time / batch_size,
                    'throughput': batch_size / batch_time
                }
        
        # Calculate statistics
        results = {
            'single_prediction_mean': float(np.mean(single_times)),
            'single_prediction_std': float(np.std(single_times)),
            'single_prediction_min': float(np.min(single_times)),
            'single_prediction_max': float(np.max(single_times)),
            'throughput_single': 1.0 / np.mean(single_times),
            'model_parameters': model.count_params(),
            'model_size_mb': model.count_params() * 4 / 1024 / 1024,  # Assuming float32
        }
        
        results.update({f'batch_results_{k}': v for k, v in batch_results.items()})
        
        logger.info(f"Benchmark completed for '{model_name}'")
        logger.info(f"Single prediction: {results['single_prediction_mean']:.4f}s ± {results['single_prediction_std']:.4f}s")
        logger.info(f"Throughput: {results['throughput_single']:.2f} predictions/second")
        
        return results
    
    def _preprocess_image(self, 
                         image: Union[np.ndarray, tf.Tensor], 
                         target_size: Tuple[int, int]) -> tf.Tensor:
        """Preprocess image for model input"""
        
        # Convert to tensor if numpy array
        if isinstance(image, np.ndarray):
            image = tf.constant(image)
        
        # Ensure float32
        image = tf.cast(image, tf.float32)
        
        # Handle grayscale images
        if len(image.shape) == 2:
            image = tf.expand_dims(image, -1)
            image = tf.repeat(image, 3, axis=-1)
        elif image.shape[-1] == 1:
            image = tf.repeat(image, 3, axis=-1)
        elif image.shape[-1] == 4:
            # RGBA to RGB
            image = image[:, :, :3]
        
        # Resize
        if image.shape[:2] != target_size:
            image = tf.image.resize(image, target_size)
        
        # Normalize to [0, 1]
        if tf.reduce_max(image) > 1.0:
            image = image / 255.0
        
        # Apply custom preprocessing if available
        if self.preprocessing_fn:
            image = self.preprocessing_fn(image)
        
        return image
    
    def _load_class_labels(self) -> Dict[int, str]:
        """Load class labels from configuration"""
        
        # Default plant disease classes
        default_labels = {
            0: "Corn_Cercospora_leaf_spot",
            1: "Corn_Common_rust",
            2: "Corn_healthy",
            3: "Corn_Northern_Leaf_Blight",
            4: "Potato_Early_blight",
            5: "Potato_healthy",
            6: "Potato_Late_blight",
            7: "Tomato_Bacterial_spot",
            8: "Tomato_Early_blight",
            9: "Tomato_healthy",
            10: "Tomato_Late_blight",
            11: "Tomato_Leaf_Mold",
            12: "Tomato_Septoria_leaf_spot",
            13: "Tomato_Spider_mites",
            14: "Tomato_Target_Spot",
            15: "Tomato_mosaic_virus",
            16: "Tomato_Yellow_Leaf_Curl_Virus"
        }
        
        # Try to load from config file
        try:
            config_path = Path(self.config.project_root) / "config" / "class_labels.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    loaded_labels = json.load(f)
                
                # Convert string keys to integers
                class_labels = {int(k): v for k, v in loaded_labels.items()}
                logger.info(f"Loaded {len(class_labels)} class labels from config")
                return class_labels
                
        except Exception as e:
            logger.warning(f"Could not load class labels from config: {e}")
        
        logger.info(f"Using default class labels ({len(default_labels)} classes)")
        return default_labels
    
    def save_predictions(self, 
                        results: List[PredictionResult], 
                        output_path: str,
                        format: str = "json") -> bool:
        """Save prediction results to file"""
        
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format == "json":
                # Convert results to serializable format
                serializable_results = []
                
                for result in results:
                    serializable_results.append({
                        'predicted_class': result.predicted_class,
                        'predicted_index': result.predicted_index,
                        'confidence': result.confidence,
                        'probabilities': result.probabilities.tolist(),
                        'processing_time': result.processing_time,
                        'metadata': result.metadata
                    })
                
                with open(output_path, 'w') as f:
                    json.dump({
                        'results': serializable_results,
                        'timestamp': datetime.now().isoformat(),
                        'total_predictions': len(results)
                    }, f, indent=2)
            
            elif format == "csv":
                import pandas as pd
                
                # Convert to DataFrame
                data = []
                for i, result in enumerate(results):
                    data.append({
                        'sample_id': i,
                        'predicted_class': result.predicted_class,
                        'confidence': result.confidence,
                        'processing_time': result.processing_time
                    })
                
                df = pd.DataFrame(data)
                df.to_csv(output_path, index=False)
            
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Saved {len(results)} predictions to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving predictions: {e}")
            return False
    
    def get_model_info(self, model_name: str = "default") -> Dict[str, Any]:
        """Get information about loaded model"""
        
        if model_name not in self.models:
            return {"error": f"Model '{model_name}' not loaded"}
        
        model = self.models[model_name]
        
        return {
            'model_name': model_name,
            'input_shape': model.input_shape,
            'output_shape': model.output_shape,
            'total_params': model.count_params(),
            'trainable_params': sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]),
            'layers': len(model.layers),
            'model_size_mb': model.count_params() * 4 / 1024 / 1024
        }


# Utility functions
def create_predictor(config: Config) -> Predictor:
    """Factory function for creating predictor"""
    return Predictor(config)


def load_and_predict(model_path: str, 
                    image_path: str, 
                    config: Config) -> PredictionResult:
    """Convenience function for single prediction"""
    
    predictor = Predictor(config)
    
    if not predictor.load_model(model_path):
        raise RuntimeError(f"Failed to load model from {model_path}")
    
    # Load image
    from PIL import Image
    image = Image.open(image_path)
    image_array = np.array(image)
    
    return predictor.predict_single(image_array)