"""
Model Factory for CAPSTONE-LAZARUS
==================================
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import logging
from typing import Dict, Any, Optional, Tuple, List
import numpy as np

from src.config import Config

logger = logging.getLogger(__name__)

class ModelFactory:
    """Factory for creating various ML/DL model architectures"""
    
    def __init__(self, config: Config):
        self.config = config
        self.model_config = config.model
        
    def create_model(self, 
                    architecture: str, 
                    input_shape: Tuple[int, ...],
                    num_classes: Optional[int] = None,
                    **kwargs) -> keras.Model:
        """
        Create model based on architecture name
        
        Args:
            architecture: Model architecture name
            input_shape: Input tensor shape
            num_classes: Number of output classes (for classification)
            **kwargs: Additional model-specific parameters
            
        Returns:
            Compiled Keras model
        """
        if num_classes is None:
            num_classes = self.model_config.num_classes
            
        logger.info(f"Creating {architecture} model with input shape {input_shape}")
        
        # Create base model
        if architecture.lower() == "efficient_net":
            model = self._create_efficient_net(input_shape, num_classes, **kwargs)
        elif architecture.lower() == "resnet":
            model = self._create_resnet(input_shape, num_classes, **kwargs)
        elif architecture.lower() == "mobilenet":
            model = self._create_mobilenet(input_shape, num_classes, **kwargs)
        elif architecture.lower() == "vit":
            model = self._create_vision_transformer(input_shape, num_classes, **kwargs)
        elif architecture.lower() == "custom_cnn":
            model = self._create_custom_cnn(input_shape, num_classes, **kwargs)
        elif architecture.lower() == "mlp":
            model = self._create_mlp(input_shape, num_classes, **kwargs)
        elif architecture.lower() == "lstm":
            model = self._create_lstm(input_shape, num_classes, **kwargs)
        elif architecture.lower() == "transformer":
            model = self._create_transformer(input_shape, num_classes, **kwargs)
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
        
        # Compile model
        self._compile_model(model)
        
        logger.info(f"Model created successfully. Total parameters: {model.count_params():,}")
        return model
    
    def _create_efficient_net(self, 
                             input_shape: Tuple[int, ...], 
                             num_classes: int,
                             variant: str = "B0",
                             pretrained: bool = True,
                             **kwargs) -> keras.Model:
        """Create EfficientNet model"""
        
        # Map variant to EfficientNet class
        efficient_net_map = {
            "B0": keras.applications.EfficientNetB0,
            "B1": keras.applications.EfficientNetB1,
            "B2": keras.applications.EfficientNetB2,
            "B3": keras.applications.EfficientNetB3,
            "B4": keras.applications.EfficientNetB4,
            "B5": keras.applications.EfficientNetB5,
            "B6": keras.applications.EfficientNetB6,
            "B7": keras.applications.EfficientNetB7,
        }
        
        if variant not in efficient_net_map:
            variant = "B0"
            logger.warning(f"Unknown EfficientNet variant, using {variant}")
        
        # Create base model
        weights = 'imagenet' if pretrained else None
        base_model = efficient_net_map[variant](
            weights=weights,
            include_top=False,
            input_shape=input_shape
        )
        
        # Add custom head
        inputs = base_model.input
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        
        if self.model_config.dropout_rate > 0:
            x = layers.Dropout(self.model_config.dropout_rate)(x)
        
        # Output layer
        if self.config.task == "classification":
            activation = "softmax" if num_classes > 1 else "sigmoid"
            outputs = layers.Dense(
                num_classes, 
                activation=activation,
                name="predictions",
                kernel_regularizer=keras.regularizers.l2(self.model_config.l2_regularization)
            )(x)
        else:
            outputs = layers.Dense(1, activation="linear", name="predictions")(x)
        
        model = keras.Model(inputs, outputs, name=f"EfficientNet{variant}")
        
        # Fine-tuning setup
        if pretrained:
            # Freeze base model initially
            base_model.trainable = False
        
        return model
    
    def _create_resnet(self, 
                      input_shape: Tuple[int, ...], 
                      num_classes: int,
                      variant: str = "50",
                      pretrained: bool = True,
                      **kwargs) -> keras.Model:
        """Create ResNet model"""
        
        resnet_map = {
            "50": keras.applications.ResNet50,
            "101": keras.applications.ResNet101,
            "152": keras.applications.ResNet152,
            "50V2": keras.applications.ResNet50V2,
            "101V2": keras.applications.ResNet101V2,
            "152V2": keras.applications.ResNet152V2,
        }
        
        if variant not in resnet_map:
            variant = "50"
            logger.warning(f"Unknown ResNet variant, using ResNet{variant}")
        
        weights = 'imagenet' if pretrained else None
        base_model = resnet_map[variant](
            weights=weights,
            include_top=False,
            input_shape=input_shape
        )
        
        # Add custom head
        inputs = base_model.input
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        
        if self.model_config.dropout_rate > 0:
            x = layers.Dropout(self.model_config.dropout_rate)(x)
        
        if self.config.task == "classification":
            activation = "softmax" if num_classes > 1 else "sigmoid"
            outputs = layers.Dense(num_classes, activation=activation, name="predictions")(x)
        else:
            outputs = layers.Dense(1, activation="linear", name="predictions")(x)
        
        model = keras.Model(inputs, outputs, name=f"ResNet{variant}")
        
        if pretrained:
            base_model.trainable = False
        
        return model
    
    def _create_mobilenet(self, 
                         input_shape: Tuple[int, ...], 
                         num_classes: int,
                         variant: str = "V2",
                         pretrained: bool = True,
                         **kwargs) -> keras.Model:
        """Create MobileNet model"""
        
        mobilenet_map = {
            "V1": keras.applications.MobileNet,
            "V2": keras.applications.MobileNetV2,
            "V3Small": keras.applications.MobileNetV3Small,
            "V3Large": keras.applications.MobileNetV3Large,
        }
        
        if variant not in mobilenet_map:
            variant = "V2"
            logger.warning(f"Unknown MobileNet variant, using MobileNet{variant}")
        
        weights = 'imagenet' if pretrained else None
        base_model = mobilenet_map[variant](
            weights=weights,
            include_top=False,
            input_shape=input_shape
        )
        
        # Add custom head
        inputs = base_model.input
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        
        if self.model_config.dropout_rate > 0:
            x = layers.Dropout(self.model_config.dropout_rate)(x)
        
        if self.config.task == "classification":
            activation = "softmax" if num_classes > 1 else "sigmoid"
            outputs = layers.Dense(num_classes, activation=activation, name="predictions")(x)
        else:
            outputs = layers.Dense(1, activation="linear", name="predictions")(x)
        
        model = keras.Model(inputs, outputs, name=f"MobileNet{variant}")
        
        if pretrained:
            base_model.trainable = False
        
        return model
    
    def _create_vision_transformer(self, 
                                  input_shape: Tuple[int, ...], 
                                  num_classes: int,
                                  patch_size: int = 16,
                                  num_heads: int = 8,
                                  num_layers: int = 6,
                                  hidden_dim: int = 768,
                                  **kwargs) -> keras.Model:
        """Create Vision Transformer (ViT) model"""
        
        def mlp(x, hidden_units, dropout_rate):
            for units in hidden_units:
                x = layers.Dense(units, activation=tf.nn.gelu)(x)
                x = layers.Dropout(dropout_rate)(x)
            return x
        
        def create_patches(images, patch_size):
            batch_size = tf.shape(images)[0]
            patches = tf.image.extract_patches(
                images=images,
                sizes=[1, patch_size, patch_size, 1],
                strides=[1, patch_size, patch_size, 1],
                rates=[1, 1, 1, 1],
                padding="VALID",
            )
            patch_dims = patches.shape[-1]
            patches = tf.reshape(patches, [batch_size, -1, patch_dims])
            return patches
        
        class Patches(layers.Layer):
            def __init__(self, patch_size):
                super(Patches, self).__init__()
                self.patch_size = patch_size
            
            def call(self, images):
                return create_patches(images, self.patch_size)
        
        class PatchEncoder(layers.Layer):
            def __init__(self, num_patches, projection_dim):
                super(PatchEncoder, self).__init__()
                self.num_patches = num_patches
                self.projection = layers.Dense(units=projection_dim)
                self.position_embedding = layers.Embedding(
                    input_dim=num_patches, output_dim=projection_dim
                )
            
            def call(self, patch):
                positions = tf.range(start=0, limit=self.num_patches, delta=1)
                encoded = self.projection(patch) + self.position_embedding(positions)
                return encoded
        
        # Calculate number of patches
        num_patches = (input_shape[0] // patch_size) * (input_shape[1] // patch_size)
        projection_dim = hidden_dim
        
        inputs = layers.Input(shape=input_shape)
        
        # Create patches
        patches = Patches(patch_size)(inputs)
        
        # Encode patches
        encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)
        
        # Create multiple layers of the Transformer block
        for _ in range(num_layers):
            # Layer normalization 1
            x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
            
            # Multi-head attention
            attention_output = layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=projection_dim, dropout=self.model_config.dropout_rate
            )(x1, x1)
            
            # Skip connection 1
            x2 = layers.Add()([attention_output, encoded_patches])
            
            # Layer normalization 2
            x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
            
            # MLP
            x3 = mlp(x3, hidden_units=[projection_dim * 2, projection_dim], dropout_rate=self.model_config.dropout_rate)
            
            # Skip connection 2
            encoded_patches = layers.Add()([x3, x2])
        
        # Final layer normalization
        representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        
        # Global average pooling
        representation = layers.GlobalAveragePooling1D()(representation)
        
        # Add a fully connected layer
        features = layers.Dense(projection_dim, activation="gelu")(representation)
        features = layers.Dropout(self.model_config.dropout_rate)(features)
        
        # Classify outputs
        if self.config.task == "classification":
            activation = "softmax" if num_classes > 1 else "sigmoid"
            logits = layers.Dense(num_classes, activation=activation)(features)
        else:
            logits = layers.Dense(1, activation="linear")(features)
        
        model = keras.Model(inputs=inputs, outputs=logits, name="VisionTransformer")
        return model
    
    def _create_custom_cnn(self, 
                          input_shape: Tuple[int, ...], 
                          num_classes: int,
                          **kwargs) -> keras.Model:
        """Create custom CNN architecture"""
        
        inputs = layers.Input(shape=input_shape)
        
        x = inputs
        
        # First block
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Second block
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Third block
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Global pooling and classification
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.model_config.dropout_rate)(x)
        
        if self.config.task == "classification":
            activation = "softmax" if num_classes > 1 else "sigmoid"
            outputs = layers.Dense(num_classes, activation=activation, name="predictions")(x)
        else:
            outputs = layers.Dense(1, activation="linear", name="predictions")(x)
        
        model = keras.Model(inputs, outputs, name="CustomCNN")
        return model
    
    def _create_mlp(self, 
                   input_shape: Tuple[int, ...], 
                   num_classes: int,
                   hidden_layers: List[int] = None,
                   **kwargs) -> keras.Model:
        """Create Multi-Layer Perceptron"""
        
        if hidden_layers is None:
            hidden_layers = [512, 256, 128]
        
        inputs = layers.Input(shape=input_shape)
        
        x = layers.Flatten()(inputs)
        
        for units in hidden_layers:
            x = layers.Dense(units, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(self.model_config.dropout_rate)(x)
        
        if self.config.task == "classification":
            activation = "softmax" if num_classes > 1 else "sigmoid"
            outputs = layers.Dense(num_classes, activation=activation, name="predictions")(x)
        else:
            outputs = layers.Dense(1, activation="linear", name="predictions")(x)
        
        model = keras.Model(inputs, outputs, name="MLP")
        return model
    
    def _create_lstm(self, 
                    input_shape: Tuple[int, ...], 
                    num_classes: int,
                    lstm_units: int = 128,
                    num_layers: int = 2,
                    **kwargs) -> keras.Model:
        """Create LSTM model for sequence data"""
        
        inputs = layers.Input(shape=input_shape)
        
        x = inputs
        for i in range(num_layers):
            return_sequences = i < num_layers - 1
            x = layers.LSTM(
                lstm_units, 
                return_sequences=return_sequences,
                dropout=self.model_config.dropout_rate,
                recurrent_dropout=self.model_config.dropout_rate
            )(x)
        
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(self.model_config.dropout_rate)(x)
        
        if self.config.task == "classification":
            activation = "softmax" if num_classes > 1 else "sigmoid"
            outputs = layers.Dense(num_classes, activation=activation, name="predictions")(x)
        else:
            outputs = layers.Dense(1, activation="linear", name="predictions")(x)
        
        model = keras.Model(inputs, outputs, name="LSTM")
        return model
    
    def _create_transformer(self, 
                           input_shape: Tuple[int, ...], 
                           num_classes: int,
                           num_heads: int = 8,
                           num_layers: int = 4,
                           d_model: int = 128,
                           **kwargs) -> keras.Model:
        """Create Transformer model for sequence data"""
        
        def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
            # Attention and Normalization
            x = layers.MultiHeadAttention(
                key_dim=head_size, num_heads=num_heads, dropout=dropout
            )(inputs, inputs)
            x = layers.Dropout(dropout)(x)
            res = x + inputs
            x = layers.LayerNormalization(epsilon=1e-6)(res)
            
            # Feed Forward Part
            x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
            x = layers.Dropout(dropout)(x)
            x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
            x = layers.Dropout(dropout)(x)
            return x + res
        
        inputs = layers.Input(shape=input_shape)
        
        # Positional encoding (simple)
        x = inputs
        
        # Stack transformer layers
        for _ in range(num_layers):
            x = transformer_encoder(
                x, 
                head_size=d_model // num_heads, 
                num_heads=num_heads, 
                ff_dim=4 * d_model,
                dropout=self.model_config.dropout_rate
            )
        
        x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
        
        # Final dense layers
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(self.model_config.dropout_rate)(x)
        
        if self.config.task == "classification":
            activation = "softmax" if num_classes > 1 else "sigmoid"
            outputs = layers.Dense(num_classes, activation=activation, name="predictions")(x)
        else:
            outputs = layers.Dense(1, activation="linear", name="predictions")(x)
        
        model = keras.Model(inputs, outputs, name="Transformer")
        return model
    
    def _compile_model(self, model: keras.Model):
        """Compile model with appropriate optimizer, loss, and metrics"""
        
        # Optimizer
        if self.config.training.optimizer.lower() == "adam":
            optimizer = keras.optimizers.Adam(learning_rate=self.config.training.learning_rate)
        elif self.config.training.optimizer.lower() == "sgd":
            optimizer = keras.optimizers.SGD(learning_rate=self.config.training.learning_rate, momentum=0.9)
        elif self.config.training.optimizer.lower() == "rmsprop":
            optimizer = keras.optimizers.RMSprop(learning_rate=self.config.training.learning_rate)
        else:
            optimizer = keras.optimizers.Adam(learning_rate=self.config.training.learning_rate)
        
        # Loss function
        if self.config.task == "classification":
            if self.config.model.num_classes > 1:
                loss = self.config.training.loss_function
            else:
                loss = "binary_crossentropy"
        else:
            loss = "mse"
        
        # Metrics
        metrics = self.config.training.metrics.copy()
        
        # Mixed precision
        if self.config.training.use_mixed_precision:
            # Wrap optimizer for mixed precision
            optimizer = keras.mixed_precision.LossScaleOptimizer(optimizer)
        
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        
        logger.info(f"Model compiled with optimizer: {self.config.training.optimizer}, loss: {loss}")
    
    def get_model_summary(self, model: keras.Model) -> Dict[str, Any]:
        """Get comprehensive model summary"""
        
        try:
            # Get model summary as string
            string_list = []
            model.summary(print_fn=lambda x: string_list.append(x))
            summary_string = "\n".join(string_list)
            
            # Calculate model size
            param_count = model.count_params()
            trainable_params = sum([np.prod(v.get_shape()) for v in model.trainable_weights])
            non_trainable_params = param_count - trainable_params
            
            # Estimate model size in MB (rough approximation)
            model_size_mb = (param_count * 4) / (1024 * 1024)  # 4 bytes per float32 parameter
            
            return {
                "model_name": model.name,
                "total_parameters": param_count,
                "trainable_parameters": int(trainable_params),
                "non_trainable_parameters": int(non_trainable_params),
                "estimated_size_mb": round(model_size_mb, 2),
                "input_shape": model.input_shape,
                "output_shape": model.output_shape,
                "summary": summary_string,
                "layer_count": len(model.layers),
                "architecture": self.model_config.architecture
            }
        
        except Exception as e:
            logger.error(f"Failed to generate model summary: {e}")
            return {"error": str(e)}
    
    def save_model_architecture(self, model: keras.Model, path: str):
        """Save model architecture to JSON"""
        from pathlib import Path
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save architecture
        with open(path.with_suffix('.json'), 'w') as f:
            f.write(model.to_json())
        
        # Save summary
        summary = self.get_model_summary(model)
        import json
        with open(path.with_suffix('.summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Model architecture saved to {path}")

# Factory function for easy model creation
def create_model(config: Config, architecture: str, input_shape: Tuple[int, ...], **kwargs) -> keras.Model:
    """Factory function for creating models"""
    factory = ModelFactory(config)
    return factory.create_model(architecture, input_shape, **kwargs)