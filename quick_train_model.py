#!/usr/bin/env python3
"""
Quick Model Training Script
===========================
Train a simple model for testing the pipeline.
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

try:
    import tensorflow as tf
    from tensorflow import keras
    
    print("🚀 Quick Model Training")
    print("=" * 50)
    
    # Create model directories
    models_dir = Path("models/best_models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a simple model
    model = keras.Sequential([
        keras.layers.Input(shape=(224, 224, 3)),
        keras.layers.Conv2D(32, 3, activation='relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(64, 3, activation='relu'),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(19, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"📊 Model created with {model.count_params():,} parameters")
    
    # Save model
    model_path = models_dir / "quick_trained_model.h5"
    model.save(str(model_path))
    
    print(f"✅ Model saved: {model_path}")
    print(f"💾 Size: {model_path.stat().st_size / (1024*1024):.1f} MB")
    
except ImportError as e:
    print(f"❌ Error: {e}")
    print("Please install TensorFlow to run training")
