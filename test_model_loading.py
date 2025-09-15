"""
Simple model loading script to test TensorFlow model compatibility
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Reduce TensorFlow logging

import tensorflow as tf
import numpy as np

def load_model_compatible():
    """Load model with fallback compatibility for different TensorFlow versions"""
    model_path = './inception_lazarus'
    
    try:
        # Method 1: Try direct loading (works with older TF versions)
        print("Attempting direct model loading...")
        model = tf.keras.models.load_model(model_path)
        print("✅ Model loaded successfully with direct loading")
        return model, "direct"
    
    except Exception as e1:
        print(f"Direct loading failed: {e1}")
        
        try:
            # Method 2: Use SavedModel directly
            print("Attempting SavedModel loading...")
            model = tf.saved_model.load(model_path)
            print("✅ Model loaded successfully with SavedModel")
            return model, "savedmodel"
        
        except Exception as e2:
            print(f"SavedModel loading failed: {e2}")
            
            try:
                # Method 3: Try TFSMLayer approach (Keras 3)
                print("Attempting TFSMLayer loading...")
                input_layer = tf.keras.layers.Input(shape=(256, 256, 3))
                tfsm_layer = tf.keras.layers.TFSMLayer(model_path, call_endpoint='serving_default')
                output = tfsm_layer(input_layer)
                model = tf.keras.Model(inputs=input_layer, outputs=output)
                print("✅ Model loaded successfully with TFSMLayer")
                return model, "tfsmlayer"
            
            except Exception as e3:
                print(f"TFSMLayer loading failed: {e3}")
                print("❌ All model loading methods failed")
                return None, "failed"

def test_model_prediction(model, model_type):
    """Test model prediction with sample data"""
    try:
        # Create test input
        test_input = np.random.rand(1, 256, 256, 3).astype(np.float32)
        
        if model_type == "savedmodel":
            # For SavedModel, use different prediction method
            infer = model.signatures["serving_default"]
            prediction = infer(tf.constant(test_input))
            # Extract output (might be in a dictionary)
            if isinstance(prediction, dict):
                prediction = list(prediction.values())[0]
        else:
            # For Keras models
            prediction = model.predict(test_input, verbose=0)
        
        print(f"✅ Prediction successful. Output shape: {prediction.shape}")
        return True
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return False

if __name__ == "__main__":
    print("🔄 Testing TensorFlow model loading compatibility...")
    print(f"TensorFlow version: {tf.__version__}")
    
    model, model_type = load_model_compatible()
    
    if model is not None:
        print(f"\n🔄 Testing prediction with {model_type} model...")
        success = test_model_prediction(model, model_type)
        
        if success:
            print("🎉 Model loading and prediction test successful!")
        else:
            print("⚠️ Model loaded but prediction failed")
    else:
        print("❌ Model loading test failed")