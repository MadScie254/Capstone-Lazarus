"""
Working solution for loading SavedModel with TensorFlow 2.x
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np

def create_prediction_function():
    """Create a prediction function that works with SavedModel"""
    try:
        # Load the SavedModel
        model = tf.saved_model.load('./inception_lazarus')
        
        # Get the inference function
        if hasattr(model, 'signatures'):
            # Use the serving signature
            infer_func = model.signatures['serving_default']
        elif callable(model):
            # If model is directly callable
            infer_func = model
        else:
            print("❌ Could not find inference function in model")
            return None
        
        def predict(input_image):
            """Prediction wrapper function"""
            # Ensure input is the right type and shape
            if len(input_image.shape) == 3:
                input_image = np.expand_dims(input_image, axis=0)
            
            input_tensor = tf.convert_to_tensor(input_image, dtype=tf.float32)
            
            # Get prediction
            prediction = infer_func(input_tensor)
            
            # Handle different output formats
            if isinstance(prediction, dict):
                # Extract the first (usually only) output
                key = list(prediction.keys())[0]
                result = prediction[key].numpy()
            else:
                result = prediction.numpy()
            
            return result
        
        print("✅ Prediction function created successfully")
        return predict
        
    except Exception as e:
        print(f"❌ Failed to create prediction function: {e}")
        return None

def test_prediction_function():
    """Test the prediction function"""
    predict_fn = create_prediction_function()
    
    if predict_fn is None:
        return False
    
    try:
        # Create test image
        test_image = np.random.rand(256, 256, 3).astype(np.float32)
        
        # Get prediction
        result = predict_fn(test_image)
        
        print(f"✅ Prediction successful! Output shape: {result.shape}")
        print(f"✅ Prediction values range: {result.min():.4f} to {result.max():.4f}")
        
        # Find top prediction
        top_class = np.argmax(result[0])
        confidence = result[0][top_class]
        print(f"✅ Top class: {top_class}, Confidence: {confidence:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Prediction test failed: {e}")
        return False

if __name__ == "__main__":
    print("🔄 Testing SavedModel prediction function...")
    print(f"TensorFlow version: {tf.__version__}")
    
    success = test_prediction_function()
    
    if success:
        print("🎉 SavedModel prediction test successful!")
    else:
        print("❌ SavedModel prediction test failed")