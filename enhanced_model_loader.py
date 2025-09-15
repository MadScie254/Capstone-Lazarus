"""
Enhanced Model Loading Utilities for Plant Disease Detection
This module provides comprehensive model loading with multiple format support
and fallback mechanisms for TensorFlow compatibility.
"""

import os
import json
import numpy as np
import tensorflow as tf
from typing import Optional, Dict, Any, Tuple, List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelManager:
    """Advanced model manager with ensemble support and compatibility handling"""
    
    def __init__(self, model_path: str = './models'):
        self.model_path = model_path
        self.loaded_models = {}
        self.ensemble_model = None
        self.model_metadata = {}
        
    def load_single_model(self, model_name: str, model_dir: str) -> Optional[tf.keras.Model]:
        """Load a single model with multiple format fallback"""
        
        # Try different formats in order of preference
        formats_to_try = [
            ('keras', f"{model_name}_model.keras"),
            ('savedmodel', 'savedmodel'),
            ('h5_weights', f"{model_name}_weights.h5"),
            ('legacy', model_name)  # Legacy SavedModel format
        ]
        
        for format_type, path in formats_to_try:
            full_path = os.path.join(model_dir, path)
            
            try:
                if format_type == 'keras' and os.path.exists(full_path):
                    logger.info(f"Loading {model_name} in Keras format...")
                    model = tf.keras.models.load_model(full_path)
                    logger.info(f"✅ Successfully loaded {model_name} (Keras format)")
                    return model
                    
                elif format_type == 'savedmodel' and os.path.exists(full_path):
                    logger.info(f"Loading {model_name} in SavedModel format...")
                    # Try using TFSMLayer for Keras 3 compatibility
                    try:
                        inputs = tf.keras.layers.Input(shape=(256, 256, 3))
                        tfsm_layer = tf.keras.layers.TFSMLayer(
                            full_path, 
                            call_endpoint='serving_default'
                        )
                        outputs = tfsm_layer(inputs)
                        model = tf.keras.Model(inputs=inputs, outputs=outputs)
                        logger.info(f"✅ Successfully loaded {model_name} (TFSMLayer)")
                        return model
                    except Exception:
                        # Fallback to direct SavedModel loading
                        model = tf.saved_model.load(full_path)
                        logger.info(f"✅ Successfully loaded {model_name} (SavedModel)")
                        return model
                        
                elif format_type == 'h5_weights':
                    # Load architecture and weights separately
                    architecture_path = os.path.join(model_dir, f"{model_name}_architecture.json")
                    if os.path.exists(full_path) and os.path.exists(architecture_path):
                        logger.info(f"Loading {model_name} from weights and architecture...")
                        
                        with open(architecture_path, 'r') as f:
                            model_json = f.read()
                        
                        model = tf.keras.models.model_from_json(model_json)
                        model.load_weights(full_path)
                        logger.info(f"✅ Successfully loaded {model_name} (weights + architecture)")
                        return model
                        
            except Exception as e:
                logger.warning(f"Failed to load {model_name} in {format_type} format: {e}")
                continue
        
        logger.error(f"❌ Could not load {model_name} in any format")
        return None
    
    def load_all_models(self) -> Dict[str, tf.keras.Model]:
        """Load all available models"""
        logger.info("🔄 Loading all available models...")
        
        # Load ensemble metadata if available
        ensemble_info_path = os.path.join(self.model_path, 'ensemble_info.json')
        if os.path.exists(ensemble_info_path):
            with open(ensemble_info_path, 'r') as f:
                self.model_metadata = json.load(f)
            logger.info("📊 Loaded ensemble metadata")
        
        # Load individual models
        for model_name in ['efficientnet', 'resnet', 'inception']:
            model_dir = os.path.join(self.model_path, model_name)
            if os.path.exists(model_dir):
                model = self.load_single_model(model_name, model_dir)
                if model:
                    self.loaded_models[model_name] = model
        
        logger.info(f"✅ Loaded {len(self.loaded_models)} models: {list(self.loaded_models.keys())}")
        return self.loaded_models
    
    def load_best_model(self) -> Optional[tf.keras.Model]:
        """Load the best performing model"""
        
        # Try to load the v2 model first
        best_model_paths = [
            './inception_lazarus_v2.keras',
            './inception_lazarus_v2',
            './inception_lazarus'
        ]
        
        for model_path in best_model_paths:
            try:
                if model_path.endswith('.keras') and os.path.exists(model_path):
                    logger.info(f"Loading best model from {model_path}...")
                    model = tf.keras.models.load_model(model_path)
                    logger.info(f"✅ Successfully loaded best model (Keras format)")
                    return model
                    
                elif os.path.exists(model_path):
                    logger.info(f"Loading best model from {model_path}...")
                    
                    try:
                        # Try TFSMLayer approach for compatibility
                        inputs = tf.keras.layers.Input(shape=(256, 256, 3))
                        tfsm_layer = tf.keras.layers.TFSMLayer(
                            model_path, 
                            call_endpoint='serving_default'
                        )
                        outputs = tfsm_layer(inputs)
                        model = tf.keras.Model(inputs=inputs, outputs=outputs)
                        logger.info(f"✅ Successfully loaded best model (TFSMLayer)")
                        return model
                        
                    except Exception:
                        # Fallback to SavedModel loading
                        model = tf.saved_model.load(model_path)
                        logger.info(f"✅ Successfully loaded best model (SavedModel)")
                        return model
                        
            except Exception as e:
                logger.warning(f"Failed to load from {model_path}: {e}")
                continue
        
        # If all else fails, load from individual models
        if self.loaded_models:
            best_model_name = self.model_metadata.get('best_model', list(self.loaded_models.keys())[0])
            if best_model_name in self.loaded_models:
                logger.info(f"✅ Using {best_model_name} as best model")
                return self.loaded_models[best_model_name]
        
        logger.error("❌ Could not load any model")
        return None
    
    def create_ensemble_predictor(self) -> Optional['EnsemblePredictor']:
        """Create an ensemble predictor from loaded models"""
        if len(self.loaded_models) < 2:
            logger.warning("Not enough models for ensemble. Need at least 2.")
            return None
        
        return EnsemblePredictor(self.loaded_models, self.model_metadata)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information"""
        info = {
            'available_models': list(self.loaded_models.keys()),
            'num_models': len(self.loaded_models),
            'metadata': self.model_metadata,
            'ensemble_available': len(self.loaded_models) >= 2
        }
        return info

class EnsemblePredictor:
    """Enhanced ensemble predictor with uncertainty quantification"""
    
    def __init__(self, models: Dict[str, tf.keras.Model], metadata: Dict[str, Any]):
        self.models = models
        self.metadata = metadata
        self.model_names = list(models.keys())
        
        # Set equal weights by default (can be optimized later)
        self.weights = [1.0 / len(models)] * len(models)
    
    def predict(self, images: np.ndarray, return_uncertainty: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Make ensemble predictions with optional uncertainty quantification"""
        
        predictions = []
        successful_predictions = 0
        
        for name, model in self.models.items():
            try:
                # Handle different model output formats
                pred = self._predict_single_model(model, images)
                predictions.append(pred)
                successful_predictions += 1
                
            except Exception as e:
                logger.warning(f"Prediction failed for {name}: {e}")
                continue
        
        if successful_predictions == 0:
            raise RuntimeError("All models failed to make predictions")
        
        # Calculate weighted average
        predictions_array = np.array(predictions)
        ensemble_pred = np.average(predictions_array, axis=0, 
                                  weights=self.weights[:successful_predictions])
        
        uncertainty = None
        if return_uncertainty and successful_predictions > 1:
            # Calculate prediction variance as uncertainty measure
            uncertainty = np.var(predictions_array, axis=0)
            uncertainty = np.mean(uncertainty, axis=1)  # Average across classes
        
        return ensemble_pred, uncertainty
    
    def _predict_single_model(self, model, images: np.ndarray) -> np.ndarray:
        """Handle prediction for different model formats"""
        try:
            # Standard Keras model
            pred = model.predict(images, verbose=0)
            
            # Handle TFSMLayer output (might be a dict)
            if isinstance(pred, dict):
                # Get the first (usually only) output
                pred = list(pred.values())[0]
            
            return pred
            
        except Exception as e:
            # Try callable SavedModel
            if callable(model):
                pred = model(images)
                if isinstance(pred, dict):
                    pred = list(pred.values())[0]
                return pred.numpy()
            else:
                raise e

# Utility functions for easy integration
def load_model_for_app(model_path: str = './models') -> Tuple[Optional[tf.keras.Model], Optional[EnsemblePredictor], Dict[str, Any]]:
    """
    Convenient function to load models for the Streamlit app
    Returns: (best_model, ensemble_predictor, model_info)
    """
    
    manager = ModelManager(model_path)
    
    # Load all models
    manager.load_all_models()
    
    # Get best model
    best_model = manager.load_best_model()
    
    # Create ensemble if possible
    ensemble = manager.create_ensemble_predictor()
    
    # Get model info
    model_info = manager.get_model_info()
    
    return best_model, ensemble, model_info

def predict_with_fallback(image: np.ndarray, 
                         best_model: Optional[tf.keras.Model], 
                         ensemble: Optional[EnsemblePredictor]) -> Tuple[np.ndarray, float, Optional[float]]:
    """
    Make predictions with fallback mechanisms
    Returns: (predictions, confidence, uncertainty)
    """
    
    if image.ndim == 3:
        image = np.expand_dims(image, axis=0)
    
    predictions = None
    uncertainty = None
    
    # Try ensemble first (if available)
    if ensemble:
        try:
            predictions, uncertainty = ensemble.predict(image, return_uncertainty=True)
            logger.info("✅ Ensemble prediction successful")
        except Exception as e:
            logger.warning(f"Ensemble prediction failed: {e}")
    
    # Fallback to best model
    if predictions is None and best_model:
        try:
            if hasattr(best_model, 'predict'):
                predictions = best_model.predict(image, verbose=0)
            elif callable(best_model):
                predictions = best_model(image)
                if isinstance(predictions, dict):
                    predictions = list(predictions.values())[0]
                predictions = predictions.numpy()
            
            logger.info("✅ Single model prediction successful")
            
        except Exception as e:
            logger.error(f"Single model prediction failed: {e}")
            raise RuntimeError("All prediction methods failed")
    
    if predictions is None:
        raise RuntimeError("No model available for prediction")
    
    # Calculate confidence
    confidence = float(np.max(predictions[0]))
    uncertainty_score = float(uncertainty[0]) if uncertainty is not None else None
    
    return predictions, confidence, uncertainty_score

# Class names for the plant disease detection
CLASS_NAMES = [
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___healthy',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Potato___Early_blight',
    'Potato___healthy',
    'Potato___Late_blight',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___healthy',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus'
]

if __name__ == "__main__":
    # Test the model loading
    print("🧪 Testing Enhanced Model Loading...")
    
    best_model, ensemble, model_info = load_model_for_app()
    
    print(f"📊 Model Info: {model_info}")
    
    if best_model:
        print("✅ Best model loaded successfully")
    
    if ensemble:
        print("✅ Ensemble predictor created successfully")
    
    print("🎉 Model loading test completed!")