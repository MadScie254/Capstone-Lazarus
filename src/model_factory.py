"""
CAPSTONE-LAZARUS: Model Factory
===============================
Advanced model architectures for plant disease classification with transfer learning.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras import applications
from tensorflow.keras.applications import (
    EfficientNetV2B0, EfficientNetV2B1, EfficientNetV2B2,
    ResNet50, ResNet101, ResNet152,
    MobileNetV3Small, MobileNetV3Large,
    DenseNet121, DenseNet169, DenseNet201,
    InceptionV3, InceptionResNetV2,
    Xception, NASNetMobile
)
from typing import Tuple, Optional, Dict, Any
import numpy as np

class ModelFactory:
    """Advanced model factory with state-of-the-art architectures for plant disease classification."""
    
    def __init__(self, input_shape: Tuple[int, int, int] = (224, 224, 3), 
                 num_classes: int = 19, use_mixed_precision: bool = True):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.use_mixed_precision = use_mixed_precision
        
        if use_mixed_precision:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
    
    def create_efficientnet_v2(self, variant: str = 'B0', dropout_rate: float = 0.3,
                              freeze_backbone: bool = False) -> Model:
        """Create EfficientNetV2 model - Optimized for accuracy and efficiency."""
        
        backbone_map = {
            'B0': EfficientNetV2B0,
            'B1': EfficientNetV2B1, 
            'B2': EfficientNetV2B2
        }
        
        if variant not in backbone_map:
            raise ValueError(f"Variant {variant} not supported. Choose from {list(backbone_map.keys())}")
        
        # Load pre-trained backbone
        backbone = backbone_map[variant](
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        if freeze_backbone:
            backbone.trainable = False
        else:
            # Fine-tune from this layer onwards
            fine_tune_at = len(backbone.layers) // 2
            for layer in backbone.layers[:fine_tune_at]:
                layer.trainable = False
        
        # Build model
        inputs = tf.keras.Input(shape=self.input_shape)
        x = backbone(inputs, training=True)
        
        # Advanced head
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
        
        # Dense layers with residual connection
        x1 = layers.Dense(512, activation='relu')(x)
        x1 = layers.BatchNormalization()(x1)
        x1 = layers.Dropout(dropout_rate/2)(x1)
        
        x2 = layers.Dense(256, activation='relu')(x1)
        x2 = layers.BatchNormalization()(x2)
        x2 = layers.Dropout(dropout_rate/2)(x2)
        
        # Output layer
        if self.use_mixed_precision:
            outputs = layers.Dense(self.num_classes, activation='softmax', dtype='float32')(x2)
        else:
            outputs = layers.Dense(self.num_classes, activation='softmax')(x2)
        
        model = Model(inputs, outputs, name=f'EfficientNetV2{variant}_PlantDisease')
        return model
    
    def create_resnet(self, variant: int = 50, dropout_rate: float = 0.3,
                     freeze_backbone: bool = False) -> Model:
        """Create ResNet model with advanced head."""
        
        backbone_map = {
            50: ResNet50,
            101: ResNet101,
            152: ResNet152
        }
        
        if variant not in backbone_map:
            raise ValueError(f"Variant {variant} not supported. Choose from {list(backbone_map.keys())}")
        
        # Load pre-trained backbone
        backbone = backbone_map[variant](
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        if freeze_backbone:
            backbone.trainable = False
        else:
            # Fine-tune from this layer onwards  
            fine_tune_at = len(backbone.layers) // 2
            for layer in backbone.layers[:fine_tune_at]:
                layer.trainable = False
        
        # Build model
        inputs = tf.keras.Input(shape=self.input_shape)
        x = backbone(inputs, training=True)
        
        # Advanced head with attention
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        
        # Squeeze-and-Excitation block
        se = layers.Dense(x.shape[-1] // 8, activation='relu')(x)
        se = layers.Dense(x.shape[-1], activation='sigmoid')(se)
        x = layers.Multiply()([x, se])
        
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate/2)(x)
        
        # Output layer
        if self.use_mixed_precision:
            outputs = layers.Dense(self.num_classes, activation='softmax', dtype='float32')(x)
        else:
            outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs, name=f'ResNet{variant}_PlantDisease')
        return model
    
    def create_mobilenet_v3(self, variant: str = 'Large', dropout_rate: float = 0.3,
                           freeze_backbone: bool = False) -> Model:
        """Create MobileNetV3 model - Optimized for mobile deployment."""
        
        backbone_map = {
            'Small': MobileNetV3Small,
            'Large': MobileNetV3Large
        }
        
        if variant not in backbone_map:
            raise ValueError(f"Variant {variant} not supported. Choose from {list(backbone_map.keys())}")
        
        # Load pre-trained backbone
        backbone = backbone_map[variant](
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        if freeze_backbone:
            backbone.trainable = False
        else:
            # Fine-tune from this layer onwards
            fine_tune_at = len(backbone.layers) // 2
            for layer in backbone.layers[:fine_tune_at]:
                layer.trainable = False
        
        # Build model
        inputs = tf.keras.Input(shape=self.input_shape)
        x = backbone(inputs, training=True)
        
        # Lightweight head for mobile efficiency
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
        
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate/2)(x)
        
        # Output layer
        if self.use_mixed_precision:
            outputs = layers.Dense(self.num_classes, activation='softmax', dtype='float32')(x)
        else:
            outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs, name=f'MobileNetV3{variant}_PlantDisease')
        return model
    
    def create_densenet(self, variant: int = 121, dropout_rate: float = 0.3,
                       freeze_backbone: bool = False) -> Model:
        """Create DenseNet model with feature reuse."""
        
        backbone_map = {
            121: DenseNet121,
            169: DenseNet169,
            201: DenseNet201
        }
        
        if variant not in backbone_map:
            raise ValueError(f"Variant {variant} not supported. Choose from {list(backbone_map.keys())}")
        
        # Load pre-trained backbone
        backbone = backbone_map[variant](
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        if freeze_backbone:
            backbone.trainable = False
        else:
            # Fine-tune from this layer onwards
            fine_tune_at = len(backbone.layers) // 2
            for layer in backbone.layers[:fine_tune_at]:
                layer.trainable = False
        
        # Build model
        inputs = tf.keras.Input(shape=self.input_shape)
        x = backbone(inputs, training=True)
        
        # Advanced head
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
        
        # Multi-scale feature fusion
        x1 = layers.Dense(512, activation='relu')(x)
        x1 = layers.BatchNormalization()(x1)
        
        x2 = layers.Dense(256, activation='relu')(x)
        x2 = layers.BatchNormalization()(x2)
        
        x_fused = layers.Concatenate()([x1, x2])
        x_fused = layers.Dropout(dropout_rate/2)(x_fused)
        
        # Output layer
        if self.use_mixed_precision:
            outputs = layers.Dense(self.num_classes, activation='softmax', dtype='float32')(x_fused)
        else:
            outputs = layers.Dense(self.num_classes, activation='softmax')(x_fused)
        
        model = Model(inputs, outputs, name=f'DenseNet{variant}_PlantDisease')
        return model
    
    def create_custom_cnn(self, dropout_rate: float = 0.3) -> Model:
        """Create custom CNN optimized for plant disease patterns."""
        
        inputs = tf.keras.Input(shape=self.input_shape)
        
        # Feature extraction with progressive complexity
        x = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Dropout(0.1)(x)
        
        x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Dropout(0.1)(x)
        
        x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Dropout(0.2)(x)
        
        x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Dropout(0.2)(x)
        
        # Global pooling and classification
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
        
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate/2)(x)
        
        # Output layer
        if self.use_mixed_precision:
            outputs = layers.Dense(self.num_classes, activation='softmax', dtype='float32')(x)
        else:
            outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs, name='CustomCNN_PlantDisease')
        return model
    
    def create_vision_transformer(self, patch_size: int = 16, num_heads: int = 8,
                                 transformer_layers: int = 6, dropout_rate: float = 0.1) -> Model:
        """Create Vision Transformer for plant disease classification."""
        
        # Calculate patches
        num_patches = (self.input_shape[0] // patch_size) ** 2
        projection_dim = 64
        
        inputs = tf.keras.Input(shape=self.input_shape)
        
        # Create patches
        patches = layers.Conv2D(projection_dim, patch_size, strides=patch_size)(inputs)
        patch_dims = patches.shape[-1]
        patches = layers.Reshape((num_patches, patch_dims))(patches)
        
        # Add position embeddings
        positions = tf.range(start=0, limit=num_patches, delta=1)
        position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)(positions)
        encoded = patches + position_embedding
        
        # Transformer blocks
        for _ in range(transformer_layers):
            # Multi-head attention
            x1 = layers.LayerNormalization(epsilon=1e-6)(encoded)
            attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim)(x1, x1)
            x2 = layers.Add()([attention, encoded])
            
            # MLP
            x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
            x3 = layers.Dense(projection_dim * 2, activation='gelu')(x3)
            x3 = layers.Dropout(dropout_rate)(x3)
            x3 = layers.Dense(projection_dim)(x3)
            x3 = layers.Dropout(dropout_rate)(x3)
            
            encoded = layers.Add()([x3, x2])
        
        # Classification head
        representation = layers.LayerNormalization(epsilon=1e-6)(encoded)
        representation = layers.GlobalAveragePooling1D()(representation)
        representation = layers.Dropout(dropout_rate)(representation)
        
        features = layers.Dense(256, activation='gelu')(representation)
        features = layers.Dropout(dropout_rate)(features)
        
        # Output layer
        if self.use_mixed_precision:
            outputs = layers.Dense(self.num_classes, activation='softmax', dtype='float32')(features)
        else:
            outputs = layers.Dense(self.num_classes, activation='softmax')(features)
        
        model = Model(inputs, outputs, name='VisionTransformer_PlantDisease')
        return model
    
    def create_hybrid_model(self, dropout_rate: float = 0.3) -> Model:
        """Create hybrid CNN-Transformer model for plant disease classification."""
        
        inputs = tf.keras.Input(shape=self.input_shape)
        
        # CNN feature extraction
        x = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(2)(x)
        
        x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(2)(x)
        
        x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        # Reshape for transformer - get static shape for reshape
        _, h, w, c = x.shape
        x = layers.Reshape((h * w, c))(x)
        
        # Simple transformer block
        x1 = layers.LayerNormalization()(x)
        attention = layers.MultiHeadAttention(num_heads=8, key_dim=256//8)(x1, x1)
        x2 = layers.Add()([attention, x])
        
        # Global pooling and classification
        x = layers.GlobalAveragePooling1D()(x2)
        x = layers.Dropout(dropout_rate)(x)
        
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate/2)(x)
        
        # Output layer
        if self.use_mixed_precision:
            outputs = layers.Dense(self.num_classes, activation='softmax', dtype='float32')(x)
        else:
            outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs, name='HybridCNNTransformer_PlantDisease')
        return model
    
    def get_model(self, architecture: str, **kwargs) -> Model:
        """Factory method to create models by architecture name."""
        
        arch_map = {
            'efficientnet_v2_b0': lambda: self.create_efficientnet_v2('B0', **kwargs),
            'efficientnet_v2_b1': lambda: self.create_efficientnet_v2('B1', **kwargs),
            'efficientnet_v2_b2': lambda: self.create_efficientnet_v2('B2', **kwargs),
            'resnet50': lambda: self.create_resnet(50, **kwargs),
            'resnet101': lambda: self.create_resnet(101, **kwargs),
            'resnet152': lambda: self.create_resnet(152, **kwargs),
            'mobilenet_v3_small': lambda: self.create_mobilenet_v3('Small', **kwargs),
            'mobilenet_v3_large': lambda: self.create_mobilenet_v3('Large', **kwargs),
            'densenet121': lambda: self.create_densenet(121, **kwargs),
            'densenet169': lambda: self.create_densenet(169, **kwargs),
            'densenet201': lambda: self.create_densenet(201, **kwargs),
            'custom_cnn': lambda: self.create_custom_cnn(**kwargs),
            'vision_transformer': lambda: self.create_vision_transformer(**kwargs),
            'hybrid_model': lambda: self.create_hybrid_model(**kwargs)
        }
        
        if architecture not in arch_map:
            raise ValueError(f"Architecture {architecture} not supported. Choose from {list(arch_map.keys())}")
        
        return arch_map[architecture]()
    
    def get_available_architectures(self) -> list:
        """Get list of available model architectures."""
        return [
            'efficientnet_v2_b0', 'efficientnet_v2_b1', 'efficientnet_v2_b2',
            'resnet50', 'resnet101', 'resnet152',
            'mobilenet_v3_small', 'mobilenet_v3_large',
            'densenet121', 'densenet169', 'densenet201',
            'custom_cnn', 'vision_transformer', 'hybrid_model'
        ]

if __name__ == "__main__":
    # Example usage
    factory = ModelFactory(input_shape=(224, 224, 3), num_classes=19)
    
    print("Available architectures:")
    for arch in factory.get_available_architectures():
        print(f"  - {arch}")
    
    # Create an EfficientNetV2 model
    model = factory.get_model('efficientnet_v2_b0', dropout_rate=0.3)
    model.summary()
