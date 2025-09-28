"""
Model Management Utilities for Lazarus Console
Handles model loading, caching, metrics, and ONNX/PyTorch switching
"""

import os
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import time
import json
import streamlit as st
from datetime import datetime

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    ort = None

class ModelManager:
    """Centralized model management with caching and performance tracking"""
    
    def __init__(self, project_root: Optional[Path] = None):
        """Initialize model manager with real model registry"""
        if project_root is None:
            self.project_root = Path(".")
        else:
            self.project_root = Path(project_root)
            
        self.models_dir = self.project_root / 'models'
        self.best_models_dir = self.models_dir / 'best_models'
        self.model_registry_path = self.models_dir / 'model_registry.json'
        
        # Load real model registry
        self.model_registry = self._load_model_registry()
        
        # Model cache and current model tracking
        self._model_cache = {}
        self._onnx_cache = {}
        self._current_model = None
        self._current_model_info = None
        
        # Performance tracking
        self._benchmark_cache = {}
        self._last_inference_time = None
    
    def _load_model_registry(self) -> Dict[str, Any]:
        """Load the real model registry from JSON file"""
        if not self.model_registry_path.exists():
            print(f"Warning: Model registry not found at {self.model_registry_path}")
            return {}
        
        try:
            with open(self.model_registry_path, 'r') as f:
                data = json.load(f)
                return data.get('models', {})
        except Exception as e:
            print(f"Error loading model registry: {e}")
            return {}
        self.registry_file = self.models_dir / 'model_registry.json'
        self.class_names_file = self.models_dir / 'class_names.json'
        
        # Cached sessions and models
        self._pytorch_cache = {}
        self._onnx_cache = {}
        self._metrics_cache = {}
        
        # Load class names
        self.class_names = self._load_class_names()
        
        # Load model registry
        self.model_registry = self._load_model_registry()
        
        # Available models
        self.available_models = self._discover_available_models()
    
    def _load_class_names(self) -> List[str]:
        """Load class names from file"""
        try:
            if self.class_names_file.exists():
                with open(self.class_names_file, 'r') as f:
                    class_names = json.load(f)
                return class_names if isinstance(class_names, list) else list(class_names.values())
            else:
                # Default plant disease classes
                return [
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
        except Exception as e:
            print(f"Error loading class names: {e}")
            return []
    
    def _load_model_registry(self) -> Dict[str, Any]:
        """Load model registry from JSON file"""
        try:
            if self.registry_file.exists():
                with open(self.registry_file, 'r') as f:
                    return json.load(f)
            else:
                return {}
        except Exception as e:
            print(f"Error loading model registry: {e}")
            return {}
    
    def _discover_available_models(self) -> List[str]:
        """Discover available model files"""
        models = []
        
        # Check best_models directory
        if self.best_models_dir.exists():
            for file_path in self.best_models_dir.iterdir():
                if file_path.suffix in ['.h5', '.keras', '.pt', '.pth', '.onnx']:
                    models.append(file_path.stem)
        
        # Also check models registry
        for model_name in self.model_registry.keys():
            if model_name not in models:
                models.append(model_name)
        
        return sorted(models)
    
    def refresh_available_models(self):
        """Refresh the list of available models"""
        self.available_models = self._discover_available_models()
        self.model_registry = self._load_model_registry()
    
    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        return self.available_models
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get detailed model information"""
        info = {
            'name': model_name,
            'exists': False,
            'size_mb': 0,
            'format': 'unknown',
            'path': None,
            'metrics': {},
            'last_modified': None
        }
        
        # Check registry first
        if model_name in self.model_registry:
            registry_info = self.model_registry[model_name]
            info.update({
                'metrics': registry_info.get('metrics', {}),
                'training_date': registry_info.get('created_at'),
                'architecture': registry_info.get('architecture', 'Unknown')
            })
        
        # Find model file
        model_path = self._find_model_path(model_name)
        if model_path and model_path.exists():
            info.update({
                'exists': True,
                'path': str(model_path),
                'size_mb': model_path.stat().st_size / (1024 * 1024),
                'format': model_path.suffix.lower(),
                'last_modified': datetime.fromtimestamp(model_path.stat().st_mtime)
            })
        
        return info
    
    def get_model_metrics(self, model_name: str) -> Dict[str, float]:
        """Get model performance metrics"""
        if model_name in self._metrics_cache:
            return self._metrics_cache[model_name]
        
        # Default metrics
        metrics = {
            'macro_f1': 0.0,
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'critical_recall': 0.0,
            'latency_ms': 0.0,
            'size_mb': 0.0,
            'calibration_ece': 1.0
        }
        
        # Get from registry
        if model_name in self.model_registry:
            registry_metrics = self.model_registry[model_name].get('metrics', {})
            metrics.update(registry_metrics)
        
        # Get file size
        model_info = self.get_model_info(model_name)
        metrics['size_mb'] = model_info.get('size_mb', 0.0)
        
        # Cache metrics
        self._metrics_cache[model_name] = metrics
        
        return metrics
    
    def _find_model_path(self, model_name: str) -> Optional[Path]:
        """Find the path to a model file"""
        
        # Check different extensions and locations
        extensions = ['.h5', '.keras', '.pt', '.pth', '.onnx']
        locations = [
            self.best_models_dir,
            self.models_dir / 'checkpoints',
            self.models_dir / 'exports',
            self.models_dir
        ]
        
        for location in locations:
            if not location.exists():
                continue
                
            for ext in extensions:
                model_path = location / f"{model_name}{ext}"
                if model_path.exists():
                    return model_path
        
        return None
    
    def load_pytorch_model(self, model_name: str, precision: str = 'fp32', 
                          amp_enabled: bool = False) -> Optional[nn.Module]:
        """Load PyTorch model with caching"""
        
        cache_key = f"{model_name}_{precision}_{'amp' if amp_enabled else 'no_amp'}"
        
        # Check cache first
        if cache_key in self._pytorch_cache:
            return self._pytorch_cache[cache_key]
        
        try:
            model_path = self._find_model_path(model_name)
            if not model_path or not model_path.exists():
                return None
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Load based on file extension
            if model_path.suffix in ['.h5', '.keras']:
                # TensorFlow/Keras model - need to convert or use alternative loading
                print(f"Warning: {model_path.suffix} files require TensorFlow. Skipping PyTorch loading.")
                return None
            
            elif model_path.suffix in ['.pt', '.pth']:
                # PyTorch model
                model = torch.load(model_path, map_location=device)
                
                if isinstance(model, dict) and 'model_state_dict' in model:
                    # State dict format - need to reconstruct model
                    # This would require knowing the model architecture
                    print(f"Warning: State dict loading requires model architecture definition")
                    return None
                
                model.eval()
                
                # Apply precision settings
                if precision == 'fp16' and device.type == 'cuda':
                    model = model.half()
                
                # Cache the loaded model
                self._pytorch_cache[cache_key] = model
                
                return model
            
            else:
                print(f"Unsupported model format: {model_path.suffix}")
                return None
                
        except Exception as e:
            print(f"Error loading PyTorch model {model_name}: {e}")
            return None
    
    def load_onnx_session(self, model_name: str) -> Optional[Any]:
        """Load ONNX inference session with caching"""
        
        if not ONNX_AVAILABLE:
            print("ONNX Runtime not available")
            return None
        
        cache_key = f"{model_name}_onnx"
        
        # Check cache first
        if cache_key in self._onnx_cache:
            return self._onnx_cache[cache_key]
        
        try:
            model_path = self._find_model_path(model_name)
            if not model_path or model_path.suffix != '.onnx':
                print(f"ONNX model not found for {model_name}")
                return None
            
            # Configure providers
            providers = ['CPUExecutionProvider']
            if torch.cuda.is_available():
                providers.insert(0, 'CUDAExecutionProvider')
            
            # Create session
            session = ort.InferenceSession(str(model_path), providers=providers)
            
            # Cache the session
            self._onnx_cache[cache_key] = session
            
            return session
            
        except Exception as e:
            print(f"Error loading ONNX session for {model_name}: {e}")
            return None
    
    def benchmark_model(self, model_name: str, num_samples: int = 100, 
                       batch_size: int = 1, precision: str = 'fp32',
                       use_onnx: bool = False) -> Dict[str, float]:
        """Benchmark model performance"""
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create dummy input
        dummy_input = torch.randn(batch_size, 3, 224, 224)
        
        if use_onnx:
            session = self.load_onnx_session(model_name)
            if not session:
                return {'error': 'ONNX session not available'}
            
            # Convert to numpy
            dummy_input_np = dummy_input.numpy()
            input_name = session.get_inputs()[0].name
            
            # Warmup
            for _ in range(10):
                _ = session.run(None, {input_name: dummy_input_np})
            
            # Benchmark
            start_time = time.time()
            for _ in range(num_samples):
                _ = session.run(None, {input_name: dummy_input_np})
            end_time = time.time()
            
        else:
            model = self.load_pytorch_model(model_name, precision, False)
            if not model:
                return {'error': 'PyTorch model not available'}
            
            model = model.to(device)
            dummy_input = dummy_input.to(device)
            
            if precision == 'fp16' and device.type == 'cuda':
                dummy_input = dummy_input.half()
            
            # Warmup
            with torch.no_grad():
                for _ in range(10):
                    _ = model(dummy_input)
            
            # Benchmark
            torch.cuda.synchronize() if device.type == 'cuda' else None
            start_time = time.time()
            
            with torch.no_grad():
                for _ in range(num_samples):
                    _ = model(dummy_input)
            
            torch.cuda.synchronize() if device.type == 'cuda' else None
            end_time = time.time()
        
        # Calculate metrics
        total_time = end_time - start_time
        avg_latency = (total_time / num_samples) * 1000  # ms
        throughput = (num_samples * batch_size) / total_time  # samples/sec
        
        return {
            'latency_ms': avg_latency,
            'throughput_sps': throughput,
            'total_time_s': total_time,
            'batch_size': batch_size,
            'precision': precision,
            'backend': 'onnx' if use_onnx else 'pytorch'
        }
    
    def clear_cache(self, model_name: str = None):
        """Clear model caches"""
        if model_name:
            # Clear specific model
            keys_to_remove = [k for k in self._pytorch_cache.keys() if model_name in k]
            for key in keys_to_remove:
                self._pytorch_cache.pop(key, None)
                
            keys_to_remove = [k for k in self._onnx_cache.keys() if model_name in k]
            for key in keys_to_remove:
                self._onnx_cache.pop(key, None)
                
            self._metrics_cache.pop(model_name, None)
        else:
            # Clear all caches
            self._pytorch_cache.clear()
            self._onnx_cache.clear()
            self._metrics_cache.clear()
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache information"""
        return {
            'pytorch_models': len(self._pytorch_cache),
            'onnx_sessions': len(self._onnx_cache),
            'cached_metrics': len(self._metrics_cache),
            'pytorch_keys': list(self._pytorch_cache.keys()),
            'onnx_keys': list(self._onnx_cache.keys())
        }