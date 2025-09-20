"""
Model Architecture Factory for Plant Disease Classification
=========================================================
Multi-architecture support with transfer learning and fine-tuning

Supported Architectures:
- EfficientNetB0 (recommended for accuracy)
- ResNet50 (robust baseline)
- MobileNetV2 (mobile deployment)
- DenseNet121 (feature reuse)
"""

import tensorflow as tf
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

def create_model(
    architecture: str,
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 19,
    dropout_rate: float = 0.3,
    fine_tune_layers: int = 50,
    weights: str = 'imagenet'
) -> tf.keras.Model:
    """
    Create model with specified architecture and transfer learning
    
    Args:
        architecture: Model architecture name
        input_shape: Input image shape (height, width, channels)
        num_classes: Number of output classes
        dropout_rate: Dropout rate for regularization
        fine_tune_layers: Number of layers to fine-tune (from the end)
        weights: Pre-trained weights ('imagenet' or None)
    
    Returns:
        Compiled Keras model ready for training
    """
    
    architecture = architecture.lower()
    logger.info(f"Creating {architecture} model with {num_classes} classes")
    
    # Create base model based on architecture
    if architecture == 'efficientnet_b0':
        base_model = create_efficientnet_b0(input_shape, weights)
    elif architecture == 'resnet50':
        base_model = create_resnet50(input_shape, weights)
    elif architecture == 'mobilenet_v2':
        base_model = create_mobilenet_v2(input_shape, weights)
    elif architecture == 'densenet121':
        base_model = create_densenet121(input_shape, weights)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Create full model with classification head
    model = tf.keras.Sequential([
        # Input layer
        tf.keras.layers.Input(shape=input_shape),
        
        # Data preprocessing (if not included in base model)
        get_preprocessing_layer(architecture),
        
        # Base model
        base_model,
        
        # Global pooling
        tf.keras.layers.GlobalAveragePooling2D(),
        
        # Dropout for regularization
        tf.keras.layers.Dropout(dropout_rate),
        
        # Dense layer for feature learning
        tf.keras.layers.Dense(
            512, 
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.L2(1e-4)
        ),
        
        # Additional dropout
        tf.keras.layers.Dropout(dropout_rate * 0.5),
        
        # Output layer
        tf.keras.layers.Dense(
            num_classes,
            activation='softmax',
            name='predictions'
        )
    ])
    
    # Configure fine-tuning
    if fine_tune_layers > 0:
        _configure_fine_tuning(base_model, fine_tune_layers)
        logger.info(f"Fine-tuning enabled for last {fine_tune_layers} layers")
    
    logger.info(f"Model created: {architecture}")
    logger.info(f"Total parameters: {model.count_params():,}")
    logger.info(f"Trainable parameters: {sum([tf.size(w) for w in model.trainable_weights]):,}")
    
    return model


def create_efficientnet_b0(input_shape: Tuple[int, int, int], weights: str) -> tf.keras.Model:
    """Create EfficientNetB0 base model"""
    return tf.keras.applications.EfficientNetB0(
        input_shape=input_shape,
        include_top=False,
        weights=weights,
        pooling=None
    )


def create_resnet50(input_shape: Tuple[int, int, int], weights: str) -> tf.keras.Model:
    """Create ResNet50 base model"""
    return tf.keras.applications.ResNet50(
        input_shape=input_shape,
        include_top=False,
        weights=weights,
        pooling=None
    )


def create_mobilenet_v2(input_shape: Tuple[int, int, int], weights: str) -> tf.keras.Model:
    """Create MobileNetV2 base model"""
    return tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights=weights,
        pooling=None,
        alpha=1.0  # Width multiplier
    )


def create_densenet121(input_shape: Tuple[int, int, int], weights: str) -> tf.keras.Model:
    """Create DenseNet121 base model"""
    return tf.keras.applications.DenseNet121(
        input_shape=input_shape,
        include_top=False,
        weights=weights,
        pooling=None
    )


def get_preprocessing_layer(architecture: str) -> tf.keras.layers.Layer:
    """Get appropriate preprocessing layer for architecture"""
    
    architecture = architecture.lower()
    
    if architecture == 'efficientnet_b0':
        # EfficientNet has built-in preprocessing
        return tf.keras.layers.Lambda(lambda x: x)
        
    elif architecture == 'resnet50':
        return tf.keras.layers.Lambda(lambda x: tf.keras.applications.resnet.preprocess_input(x))
        
    elif architecture == 'mobilenet_v2':
        return tf.keras.layers.Lambda(lambda x: tf.keras.applications.mobilenet_v2.preprocess_input(x))
        
    elif architecture == 'densenet121':
        return tf.keras.layers.Lambda(lambda x: tf.keras.applications.densenet.preprocess_input(x))
        
    else:
        # Default preprocessing (normalization to [-1, 1])
        return tf.keras.layers.Rescaling(1./127.5, offset=-1)


def _configure_fine_tuning(base_model: tf.keras.Model, fine_tune_layers: int):
    """Configure fine-tuning for base model"""
    
    # Make base model trainable
    base_model.trainable = True
    
    # Freeze early layers, fine-tune later layers
    total_layers = len(base_model.layers)
    freeze_layers = max(0, total_layers - fine_tune_layers)
    
    for i, layer in enumerate(base_model.layers):
        if i < freeze_layers:
            layer.trainable = False
        else:
            layer.trainable = True
    
    logger.info(f"Base model layers: {total_layers}")
    logger.info(f"Frozen layers: {freeze_layers}")
    logger.info(f"Fine-tuning layers: {fine_tune_layers}")


def create_ensemble_model(
    model_paths: list,
    architecture_names: list,
    ensemble_method: str = 'soft_voting',
    input_shape: Tuple[int, int, int] = (224, 224, 3)
) -> tf.keras.Model:
    """
    Create ensemble model from multiple trained models
    
    Args:
        model_paths: List of paths to trained model files
        architecture_names: List of architecture names for each model
        ensemble_method: 'soft_voting', 'hard_voting', or 'average'
        input_shape: Input shape for ensemble
    
    Returns:
        Ensemble model
    """
    
    logger.info(f"Creating ensemble model with {len(model_paths)} models")
    logger.info(f"Ensemble method: {ensemble_method}")
    
    # Load individual models
    models = []
    for i, (model_path, arch_name) in enumerate(zip(model_paths, architecture_names)):
        try:
            model = tf.keras.models.load_model(model_path)
            model._name = f"model_{i}_{arch_name}"
            models.append(model)
            logger.info(f"Loaded model {i+1}: {arch_name}")
        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {e}")
            raise
    
    if not models:
        raise ValueError("No models loaded for ensemble")
    
    # Create ensemble architecture
    input_layer = tf.keras.layers.Input(shape=input_shape)
    
    # Get outputs from all models
    model_outputs = []
    for model in models:
        # Ensure model doesn't update during ensemble inference
        for layer in model.layers:
            layer.trainable = False
        
        output = model(input_layer)
        model_outputs.append(output)
    
    # Combine outputs based on ensemble method
    if ensemble_method == 'soft_voting' or ensemble_method == 'average':
        # Average the softmax outputs
        ensemble_output = tf.keras.layers.Average()(model_outputs)
        
    elif ensemble_method == 'hard_voting':
        # Convert to hard predictions, then average
        hard_outputs = []
        for output in model_outputs:
            hard_output = tf.keras.layers.Lambda(
                lambda x: tf.cast(tf.equal(x, tf.reduce_max(x, axis=1, keepdims=True)), tf.float32)
            )(output)
            hard_outputs.append(hard_output)
        ensemble_output = tf.keras.layers.Average()(hard_outputs)
        
    else:
        raise ValueError(f"Unsupported ensemble method: {ensemble_method}")
    
    # Create final ensemble model
    ensemble_model = tf.keras.Model(
        inputs=input_layer,
        outputs=ensemble_output,
        name=f"ensemble_{ensemble_method}"
    )
    
    logger.info(f"Ensemble model created successfully")
    logger.info(f"Total parameters: {ensemble_model.count_params():,}")
    
    return ensemble_model


def get_model_summary(model: tf.keras.Model) -> dict:
    """Get comprehensive model summary statistics"""
    
    summary_stats = {
        'total_params': model.count_params(),
        'trainable_params': sum([tf.size(w) for w in model.trainable_weights]),
        'non_trainable_params': sum([tf.size(w) for w in model.non_trainable_weights]),
        'model_size_mb': model.count_params() * 4 / (1024 * 1024),  # Approximate size in MB
        'layers': len(model.layers),
        'model_name': model.name
    }
    
    return summary_stats


def compare_architectures(input_shape: Tuple[int, int, int] = (224, 224, 3)) -> dict:
    """Compare all available architectures"""
    
    architectures = ['efficientnet_b0', 'resnet50', 'mobilenet_v2', 'densenet121']
    comparison = {}
    
    logger.info("Comparing model architectures...")
    
    for arch in architectures:
        try:
            # Create model without training
            model = create_model(
                architecture=arch,
                input_shape=input_shape,
                num_classes=19,
                dropout_rate=0.3,
                fine_tune_layers=0  # No fine-tuning for comparison
            )
            
            # Get summary statistics
            stats = get_model_summary(model)
            comparison[arch] = stats
            
            # Clean up
            del model
            
        except Exception as e:
            logger.error(f"Failed to create {arch}: {e}")
            comparison[arch] = {'error': str(e)}
    
    return comparison


# Model validation and testing functions
def validate_model_creation():
    """Validate that all architectures can be created successfully"""
    
    print("🔍 MODEL ARCHITECTURE VALIDATION")
    print("=" * 50)
    
    architectures = ['efficientnet_b0', 'resnet50', 'mobilenet_v2', 'densenet121']
    input_shape = (224, 224, 3)
    
    success_count = 0
    
    for arch in architectures:
        try:
            print(f"Testing {arch}...")
            
            # Create model
            model = create_model(
                architecture=arch,
                input_shape=input_shape,
                num_classes=19,
                dropout_rate=0.3,
                fine_tune_layers=10
            )
            
            # Test forward pass
            dummy_input = tf.random.normal((1, *input_shape))
            output = model(dummy_input)
            
            # Validate output shape
            assert output.shape == (1, 19), f"Wrong output shape: {output.shape}"
            
            print(f"✅ {arch}: {model.count_params():,} parameters")
            success_count += 1
            
            # Clean up
            del model, dummy_input, output
            
        except Exception as e:
            print(f"❌ {arch}: Failed - {e}")
    
    print(f"\n🎯 VALIDATION COMPLETE")
    print(f"Successfully created {success_count}/{len(architectures)} architectures")
    
    if success_count == len(architectures):
        print("✅ All architectures are working correctly!")
    else:
        print("⚠️  Some architectures failed - check errors above")
    
    return success_count == len(architectures)


if __name__ == "__main__":
    # Run validation
    validate_model_creation()
    
    # Show architecture comparison
    print("\n📊 ARCHITECTURE COMPARISON")
    print("=" * 50)
    comparison = compare_architectures()
    
    for arch, stats in comparison.items():
        if 'error' not in stats:
            print(f"{arch}:")
            print(f"  - Parameters: {stats['total_params']:,}")
            print(f"  - Model size: {stats['model_size_mb']:.1f} MB")
            print(f"  - Layers: {stats['layers']}")
        else:
            print(f"{arch}: ERROR - {stats['error']}")