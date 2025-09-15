import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import streamlit as st

class GradCAM:
    """Gradient-weighted Class Activation Mapping for model interpretability"""
    
    def __init__(self, model, layer_name=None):
        self.model = model
        self.layer_name = layer_name or self.find_target_layer()
        
    def find_target_layer(self):
        """Find the last convolutional layer"""
        for layer in reversed(self.model.layers):
            if 'conv' in layer.name.lower():
                return layer.name
        return None
    
    def generate_heatmap(self, image, class_idx, alpha=0.4):
        """Generate GradCAM heatmap"""
        try:
            # Create a model that maps the input image to the activations of the last conv layer
            grad_model = tf.keras.models.Model(
                [self.model.inputs], 
                [self.model.get_layer(self.layer_name).output, self.model.output]
            )
            
            # Compute gradients
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(image)
                loss = predictions[:, class_idx]
            
            # Get gradients of the loss with respect to the conv layer
            grads = tape.gradient(loss, conv_outputs)
            
            # Get the mean gradient for each feature map
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
            # Weight the conv outputs by the gradients
            conv_outputs = conv_outputs[0]
            heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)
            
            # Normalize the heatmap
            heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
            
            return heatmap.numpy()
            
        except Exception as e:
            st.error(f"Error generating heatmap: {str(e)}")
            return None
    
    def overlay_heatmap(self, image, heatmap, alpha=0.4):
        """Overlay heatmap on original image"""
        try:
            # Convert image to numpy if needed
            if isinstance(image, Image.Image):
                image = np.array(image)
            
            # Resize heatmap to match image dimensions
            heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
            
            # Convert heatmap to RGB
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            
            # Overlay the heatmap on the image
            overlaid = cv2.addWeighted(image, 1-alpha, heatmap, alpha, 0)
            
            return overlaid
            
        except Exception as e:
            st.error(f"Error overlaying heatmap: {str(e)}")
            return image

def create_feature_importance_plot(predictions, class_info):
    """Create a feature importance visualization"""
    top_5_indices = np.argsort(predictions[0])[::-1][:5]
    top_5_scores = predictions[0][top_5_indices]
    top_5_classes = [class_info[idx]['status'] for idx in top_5_indices]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(top_5_classes, top_5_scores)
    ax.set_xlabel('Prediction Confidence')
    ax.set_title('Top 5 Class Predictions')
    
    # Color bars based on confidence
    for i, bar in enumerate(bars):
        if top_5_scores[i] > 0.5:
            bar.set_color('green')
        elif top_5_scores[i] > 0.3:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    plt.tight_layout()
    return fig

def batch_predict(model, images, class_info):
    """Process multiple images at once"""
    results = []
    
    for i, image in enumerate(images):
        try:
            # Preprocess image
            processed_image = preprocess_single_image(image)
            
            # Make prediction
            prediction = model.predict(processed_image, verbose=0)
            predicted_class = np.argmax(prediction)
            confidence = float(prediction[0][predicted_class])
            
            result = {
                'image_index': i,
                'predicted_class': predicted_class,
                'confidence': confidence,
                'class_info': class_info[predicted_class],
                'all_predictions': prediction[0]
            }
            results.append(result)
            
        except Exception as e:
            st.error(f"Error processing image {i+1}: {str(e)}")
            continue
    
    return results

def preprocess_single_image(image, target_size=(256, 256)):
    """Preprocess a single image for prediction"""
    try:
        # Resize image
        if isinstance(image, Image.Image):
            image = image.resize(target_size, Image.Resampling.LANCZOS)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            img_array = np.asarray(image, dtype=np.float32)
        else:
            img_array = cv2.resize(image, target_size)
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        
        # Normalize
        img_array = img_array / 255.0
        
        # Add batch dimension
        img_batch = np.expand_dims(img_array, axis=0)
        
        return img_batch
        
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None

def calculate_model_metrics(predictions, true_labels):
    """Calculate various model performance metrics"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    
    # Convert predictions to class indices
    pred_classes = np.argmax(predictions, axis=1)
    
    metrics = {
        'accuracy': accuracy_score(true_labels, pred_classes),
        'precision': precision_score(true_labels, pred_classes, average='weighted'),
        'recall': recall_score(true_labels, pred_classes, average='weighted'),
        'f1_score': f1_score(true_labels, pred_classes, average='weighted'),
        'confusion_matrix': confusion_matrix(true_labels, pred_classes)
    }
    
    return metrics

def create_confusion_matrix_plot(cm, class_names):
    """Create a confusion matrix visualization"""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    import seaborn as sns
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.set_title('Confusion Matrix')
    
    plt.tight_layout()
    return fig

def get_plant_care_recommendations(disease_status, plant_type):
    """Provide detailed care recommendations based on disease and plant type"""
    
    recommendations = {
        'Healthy': {
            'general': "Continue regular monitoring and good agricultural practices.",
            'specific': {
                'Corn (maize)': "Maintain proper spacing, adequate fertilization, and monitor for early signs of disease.",
                'Potato': "Ensure good drainage, avoid overhead watering, and practice crop rotation.",
                'Tomato': "Provide adequate support, maintain consistent watering, and ensure good air circulation."
            }
        },
        'Bacterial Spot': {
            'general': "Apply copper-based bactericides and improve cultural practices.",
            'prevention': "Avoid overhead irrigation, use drip irrigation, sanitize tools between plants.",
            'treatment': "Apply copper fungicides every 7-14 days, remove severely affected plants."
        },
        'Early Blight': {
            'general': "Apply fungicides and improve plant spacing for better air circulation.",
            'prevention': "Mulch around plants, avoid wetting foliage, remove plant debris.",
            'treatment': "Use chlorothalonil or copper-based fungicides, remove affected lower leaves."
        },
        'Late Blight': {
            'general': "Immediate fungicide application is critical. This is a serious disease.",
            'prevention': "Plant resistant varieties, ensure good drainage, avoid overhead watering.",
            'treatment': "Apply preventive fungicides like mancozeb or chlorothalonil. Remove infected plants immediately."
        },
        # Add more disease-specific recommendations as needed
    }
    
    general_rec = recommendations.get(disease_status, {}).get('general', "Consult with agricultural extension services for specific treatment recommendations.")
    specific_rec = recommendations.get(disease_status, {}).get('specific', {}).get(plant_type, "")
    
    return {
        'general': general_rec,
        'specific': specific_rec,
        'prevention': recommendations.get(disease_status, {}).get('prevention', ""),
        'treatment': recommendations.get(disease_status, {}).get('treatment', "")
    }