"""
LoRA (Low-Rank Adaptation) and Model Adaptation for CAPSTONE-LAZARUS
===================================================================
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import logging
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

class LoRALayer(layers.Layer):
    """LoRA (Low-Rank Adaptation) layer implementation"""
    
    def __init__(self, 
                 original_layer: layers.Layer,
                 rank: int = 16,
                 alpha: float = 16.0,
                 dropout_rate: float = 0.0,
                 **kwargs):
        """
        Initialize LoRA layer
        
        Args:
            original_layer: The original layer to adapt
            rank: Rank of the adaptation
            alpha: Scaling factor
            dropout_rate: Dropout rate for LoRA matrices
        """
        super(LoRALayer, self).__init__(**kwargs)
        
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.dropout_rate = dropout_rate
        
        # Freeze original layer
        self.original_layer.trainable = False
        
        # Get original layer dimensions
        if hasattr(original_layer, 'units'):
            # Dense layer
            self.input_dim = original_layer.input_spec.axes[-1]
            self.output_dim = original_layer.units
        elif hasattr(original_layer, 'filters'):
            # Conv layer
            self.input_dim = original_layer.input_spec.axes[-1]
            self.output_dim = original_layer.filters
        else:
            raise ValueError(f"Unsupported layer type for LoRA: {type(original_layer)}")
        
        # Ensure rank is valid
        self.rank = min(rank, min(self.input_dim, self.output_dim))
        
        # LoRA matrices
        self.lora_A = None
        self.lora_B = None
        self.dropout = layers.Dropout(dropout_rate) if dropout_rate > 0 else None
        
        self.built = False
    
    def build(self, input_shape):
        """Build LoRA matrices"""
        if not self.built:
            # LoRA A matrix (input_dim x rank)
            self.lora_A = self.add_weight(
                name='lora_A',
                shape=(self.input_dim, self.rank),
                initializer='random_normal',
                trainable=True
            )
            
            # LoRA B matrix (rank x output_dim) - initialized to zero
            self.lora_B = self.add_weight(
                name='lora_B',
                shape=(self.rank, self.output_dim),
                initializer='zeros',
                trainable=True
            )
            
            self.built = True
        
        super(LoRALayer, self).build(input_shape)
    
    def call(self, inputs, training=None):
        """Forward pass through LoRA layer"""
        # Original layer output
        original_output = self.original_layer(inputs, training=training)
        
        # LoRA adaptation
        if isinstance(self.original_layer, layers.Dense):
            # For Dense layers
            lora_output = tf.matmul(inputs, self.lora_A)
            if self.dropout and training:
                lora_output = self.dropout(lora_output, training=training)
            lora_output = tf.matmul(lora_output, self.lora_B)
            
        elif isinstance(self.original_layer, layers.Conv2D):
            # For Conv2D layers (simplified - treats as dense)
            # Full implementation would handle convolution properly
            batch_size = tf.shape(inputs)[0]
            height = tf.shape(inputs)[1]
            width = tf.shape(inputs)[2]
            
            # Reshape input for matrix multiplication
            inputs_reshaped = tf.reshape(inputs, [-1, self.input_dim])
            
            # Apply LoRA
            lora_output = tf.matmul(inputs_reshaped, self.lora_A)
            if self.dropout and training:
                lora_output = self.dropout(lora_output, training=training)
            lora_output = tf.matmul(lora_output, self.lora_B)
            
            # Reshape back
            lora_output = tf.reshape(lora_output, [batch_size, height, width, self.output_dim])
        
        else:
            lora_output = 0  # No adaptation for unsupported layer types
        
        # Scale and add adaptation
        scaling = self.alpha / self.rank
        return original_output + scaling * lora_output
    
    def get_config(self):
        config = super(LoRALayer, self).get_config()
        config.update({
            'rank': self.rank,
            'alpha': self.alpha,
            'dropout_rate': self.dropout_rate
        })
        return config

class LoRAAdapter:
    """LoRA adapter for existing models"""
    
    def __init__(self, 
                 rank: int = 16,
                 alpha: float = 16.0,
                 dropout_rate: float = 0.0,
                 target_layers: Optional[List[str]] = None):
        """
        Initialize LoRA adapter
        
        Args:
            rank: Rank for LoRA adaptation
            alpha: Scaling factor
            dropout_rate: Dropout rate
            target_layers: List of layer names/types to adapt
        """
        self.rank = rank
        self.alpha = alpha
        self.dropout_rate = dropout_rate
        self.target_layers = target_layers or ['Dense', 'Conv2D']
        
    def adapt_model(self, model: keras.Model, freeze_base: bool = True) -> keras.Model:
        """
        Apply LoRA adaptation to a model
        
        Args:
            model: Original model to adapt
            freeze_base: Whether to freeze the base model
            
        Returns:
            LoRA-adapted model
        """
        logger.info(f"Applying LoRA adaptation with rank={self.rank}")
        
        # Clone the model architecture
        model_config = model.get_config()
        
        # Create new model with LoRA layers
        adapted_model = self._create_lora_model(model)
        
        # Copy weights from original model
        self._copy_weights(model, adapted_model, freeze_base)
        
        logger.info(f"LoRA adaptation complete. Trainable parameters: {adapted_model.count_params():,}")
        
        return adapted_model
    
    def _create_lora_model(self, original_model: keras.Model) -> keras.Model:
        """Create model with LoRA layers"""
        
        # Get the model's functional API structure
        if hasattr(original_model, '_layers'):
            # Sequential model
            return self._adapt_sequential_model(original_model)
        else:
            # Functional model
            return self._adapt_functional_model(original_model)
    
    def _adapt_sequential_model(self, model: keras.Model) -> keras.Model:
        """Adapt a sequential model"""
        new_layers = []
        
        for layer in model.layers:
            if self._should_adapt_layer(layer):
                # Wrap with LoRA
                lora_layer = LoRALayer(
                    layer,
                    rank=self.rank,
                    alpha=self.alpha,
                    dropout_rate=self.dropout_rate
                )
                new_layers.append(lora_layer)
            else:
                new_layers.append(layer)
        
        # Create new sequential model
        new_model = keras.Sequential(new_layers, name=f"{model.name}_LoRA")
        
        return new_model
    
    def _adapt_functional_model(self, model: keras.Model) -> keras.Model:
        """Adapt a functional model (simplified implementation)"""
        
        # For simplicity, we'll adapt the last few layers
        # Full implementation would need to traverse the graph properly
        
        inputs = model.input
        x = inputs
        
        # Get intermediate layers
        for i, layer in enumerate(model.layers[:-1]):  # Exclude output layer
            if self._should_adapt_layer(layer) and i >= len(model.layers) - 5:  # Adapt last 5 adaptable layers
                # Create LoRA layer
                lora_layer = LoRALayer(
                    layer,
                    rank=self.rank,
                    alpha=self.alpha,
                    dropout_rate=self.dropout_rate
                )
                x = lora_layer(x)
            else:
                # Use original layer
                if hasattr(layer, '__call__'):
                    x = layer(x)
        
        # Final layer (usually not adapted)
        outputs = model.layers[-1](x)
        
        new_model = keras.Model(inputs, outputs, name=f"{model.name}_LoRA")
        
        return new_model
    
    def _should_adapt_layer(self, layer: layers.Layer) -> bool:
        """Check if layer should be adapted with LoRA"""
        layer_type = layer.__class__.__name__
        
        # Check if layer type is in target layers
        if layer_type not in self.target_layers:
            return False
        
        # Additional checks
        if isinstance(layer, layers.Dense):
            return layer.units > self.rank  # Only adapt if layer is large enough
        elif isinstance(layer, layers.Conv2D):
            return layer.filters > self.rank
        
        return False
    
    def _copy_weights(self, 
                     original_model: keras.Model, 
                     adapted_model: keras.Model, 
                     freeze_base: bool):
        """Copy weights from original to adapted model"""
        
        # Build both models with same input
        dummy_input = tf.random.normal((1,) + original_model.input_shape[1:])
        original_model(dummy_input)
        adapted_model(dummy_input)
        
        # Copy weights layer by layer
        original_layers = {layer.name: layer for layer in original_model.layers}
        
        for layer in adapted_model.layers:
            if isinstance(layer, LoRALayer):
                # LoRA layer - copy original layer weights
                original_layer_name = layer.original_layer.name
                if original_layer_name in original_layers:
                    original_layer = original_layers[original_layer_name]
                    layer.original_layer.set_weights(original_layer.get_weights())
                    
                    if freeze_base:
                        layer.original_layer.trainable = False
                
                # LoRA matrices are initialized randomly/zeros, so no copying needed
            else:
                # Regular layer - copy weights if matching layer exists
                if layer.name in original_layers:
                    original_layer = original_layers[layer.name]
                    if layer.get_weights() and original_layer.get_weights():
                        layer.set_weights(original_layer.get_weights())
                    
                    if freeze_base:
                        layer.trainable = False

class AdapterLayer(layers.Layer):
    """Simple adapter layer (alternative to LoRA)"""
    
    def __init__(self, 
                 bottleneck_dim: int = 64,
                 activation: str = 'relu',
                 dropout_rate: float = 0.1,
                 **kwargs):
        super(AdapterLayer, self).__init__(**kwargs)
        
        self.bottleneck_dim = bottleneck_dim
        self.activation = activation
        self.dropout_rate = dropout_rate
        
        self.down_project = None
        self.up_project = None
        self.activation_layer = None
        self.dropout = None
        
    def build(self, input_shape):
        input_dim = input_shape[-1]
        
        # Down projection
        self.down_project = layers.Dense(
            self.bottleneck_dim,
            use_bias=False,
            kernel_initializer='he_normal'
        )
        
        # Up projection
        self.up_project = layers.Dense(
            input_dim,
            use_bias=False,
            kernel_initializer='zeros'  # Initialize to zero for stability
        )
        
        # Activation
        if self.activation:
            self.activation_layer = layers.Activation(self.activation)
        
        # Dropout
        if self.dropout_rate > 0:
            self.dropout = layers.Dropout(self.dropout_rate)
        
        super(AdapterLayer, self).build(input_shape)
    
    def call(self, inputs, training=None):
        # Residual connection
        residual = inputs
        
        # Adapter transformation
        x = self.down_project(inputs)
        
        if self.activation_layer:
            x = self.activation_layer(x)
        
        if self.dropout and training:
            x = self.dropout(x, training=training)
        
        x = self.up_project(x)
        
        # Add residual connection
        return residual + x
    
    def get_config(self):
        config = super(AdapterLayer, self).get_config()
        config.update({
            'bottleneck_dim': self.bottleneck_dim,
            'activation': self.activation,
            'dropout_rate': self.dropout_rate
        })
        return config

class ModelAdapter:
    """General model adaptation utilities"""
    
    @staticmethod
    def add_adapters(model: keras.Model, 
                    adapter_dim: int = 64,
                    insertion_points: Optional[List[str]] = None) -> keras.Model:
        """
        Add adapter layers to existing model
        
        Args:
            model: Original model
            adapter_dim: Adapter bottleneck dimension
            insertion_points: Layer names where to insert adapters
            
        Returns:
            Model with adapters
        """
        if insertion_points is None:
            # Insert adapters after each Dense/Conv layer
            insertion_points = []
            for layer in model.layers:
                if isinstance(layer, (layers.Dense, layers.Conv2D)):
                    insertion_points.append(layer.name)
        
        # Create new model with adapters
        # This is a simplified implementation
        logger.info(f"Adding adapters at {len(insertion_points)} locations")
        
        # For now, return original model
        # Full implementation would require graph reconstruction
        return model
    
    @staticmethod
    def get_adaptation_summary(model: keras.Model) -> Dict[str, Any]:
        """Get summary of model adaptations"""
        
        total_params = model.count_params()
        trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        
        # Count LoRA layers
        lora_layers = sum(1 for layer in model.layers if isinstance(layer, LoRALayer))
        
        # Count adapter layers
        adapter_layers = sum(1 for layer in model.layers if isinstance(layer, AdapterLayer))
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'frozen_parameters': total_params - trainable_params,
            'trainable_ratio': trainable_params / total_params,
            'lora_layers': lora_layers,
            'adapter_layers': adapter_layers
        }
    
    @staticmethod
    def save_lora_weights(model: keras.Model, path: str):
        """Save only LoRA weights"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        lora_weights = {}
        
        for layer in model.layers:
            if isinstance(layer, LoRALayer):
                layer_weights = {
                    'lora_A': layer.lora_A.numpy(),
                    'lora_B': layer.lora_B.numpy()
                }
                lora_weights[layer.name] = layer_weights
        
        np.savez(path.with_suffix('.npz'), **lora_weights)
        logger.info(f"LoRA weights saved to {path}")
    
    @staticmethod
    def load_lora_weights(model: keras.Model, path: str):
        """Load LoRA weights"""
        path = Path(path)
        
        if not path.exists():
            logger.error(f"LoRA weights file not found: {path}")
            return
        
        lora_weights = np.load(path.with_suffix('.npz'), allow_pickle=True)
        
        for layer in model.layers:
            if isinstance(layer, LoRALayer) and layer.name in lora_weights:
                layer_data = lora_weights[layer.name].item()
                layer.lora_A.assign(layer_data['lora_A'])
                layer.lora_B.assign(layer_data['lora_B'])
        
        logger.info(f"LoRA weights loaded from {path}")

# Utility functions
def apply_lora_adaptation(model: keras.Model, 
                         rank: int = 16,
                         alpha: float = 16.0,
                         target_layers: Optional[List[str]] = None) -> keras.Model:
    """Apply LoRA adaptation to a model"""
    adapter = LoRAAdapter(rank=rank, alpha=alpha, target_layers=target_layers)
    return adapter.adapt_model(model)