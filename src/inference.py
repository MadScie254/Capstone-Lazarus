"""
CAPSTONE-LAZARUS: Inference Engine
==================================
Advanced inference pipeline with uncertainty estimation and explainable AI.
"""

import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
import pandas as pd
from dataclasses import dataclass
import json

@dataclass
class PredictionResult:
    """Structured prediction result with confidence and explanations."""
    class_name: str
    class_index: int
    confidence: float
    all_probabilities: Dict[str, float]
    uncertainty: float
    explanation_map: Optional[np.ndarray] = None
    recommendations: Optional[List[str]] = None
    risk_level: str = "medium"

class PlantDiseaseInference:
    """Advanced inference engine for plant disease classification with explainable AI."""
    
    def __init__(self, model_path: str, class_names: List[str], 
                 img_size: Tuple[int, int] = (224, 224)):
        """
        Initialize inference engine.
        
        Args:
            model_path: Path to trained model (.h5 file)
            class_names: List of class names in order
            img_size: Input image size (height, width)
        """
        self.model_path = Path(model_path)
        self.class_names = class_names
        self.img_size = img_size
        self.num_classes = len(class_names)
        
        # Load model
        self.model = self._load_model()
        
        # Agricultural recommendations database
        self.disease_recommendations = self._load_disease_recommendations()
        
        print(f"✅ Inference engine initialized")
        print(f"   📄 Model: {self.model_path.name}")
        print(f"   🏷️  Classes: {self.num_classes}")
        print(f"   🖼️ Input size: {self.img_size}")
    
    def _load_model(self) -> tf.keras.Model:
        """Load trained model with custom objects."""
        try:
            # Define custom objects for model loading
            custom_objects = {
                'F1Score': self._get_f1_metric()
            }
            
            model = tf.keras.models.load_model(str(self.model_path), custom_objects=custom_objects)
            print(f"✅ Model loaded successfully: {model.name}")
            return model
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def _get_f1_metric(self):
        """Get F1 score metric class for model loading."""
        class F1Score(tf.keras.metrics.Metric):
            def __init__(self, name='f1_score', **kwargs):
                super().__init__(name=name, **kwargs)
                self.precision = tf.keras.metrics.Precision()
                self.recall = tf.keras.metrics.Recall()
            
            def update_state(self, y_true, y_pred, sample_weight=None):
                self.precision.update_state(y_true, y_pred, sample_weight)
                self.recall.update_state(y_true, y_pred, sample_weight)
            
            def result(self):
                p = self.precision.result()
                r = self.recall.result()
                return 2 * ((p * r) / (p + r + tf.keras.backend.epsilon()))
            
            def reset_state(self):
                self.precision.reset_state()
                self.recall.reset_state()
        
        return F1Score
    
    def _load_disease_recommendations(self) -> Dict[str, Dict[str, Any]]:
        """Load disease-specific recommendations and treatment information."""
        return {
            "corn_(maize)___healthy": {
                "description": "Healthy corn plant with no visible disease symptoms",
                "risk_level": "low",
                "actions": [
                    "Continue regular monitoring",
                    "Maintain proper irrigation",
                    "Apply preventive fungicides if weather conditions favor disease"
                ],
                "urgency": "low"
            },
            "corn_(maize)___cercospora_leaf_spot_gray_leaf_spot": {
                "description": "Fungal disease causing rectangular lesions on corn leaves",
                "risk_level": "high",
                "actions": [
                    "Apply fungicide immediately (strobilurins or triazoles)",
                    "Improve air circulation by reducing plant density",
                    "Remove infected plant debris",
                    "Consider resistant varieties for next season"
                ],
                "urgency": "high"
            },
            "corn_(maize)___common_rust_": {
                "description": "Fungal disease with orange-brown pustules on leaves",
                "risk_level": "medium",
                "actions": [
                    "Apply fungicide if infection is severe",
                    "Monitor weather conditions (high humidity favors disease)",
                    "Ensure adequate plant nutrition",
                    "Remove infected lower leaves"
                ],
                "urgency": "medium"
            },
            "corn_(maize)___northern_leaf_blight": {
                "description": "Fungal disease causing long, elliptical gray-green lesions",
                "risk_level": "high",
                "actions": [
                    "Apply fungicide treatment (QoI or DMI fungicides)",
                    "Improve field drainage",
                    "Use resistant corn hybrids",
                    "Rotate with non-host crops"
                ],
                "urgency": "high"
            },
            "potato___healthy": {
                "description": "Healthy potato plant with no disease symptoms",
                "risk_level": "low",
                "actions": [
                    "Continue regular field monitoring",
                    "Maintain proper soil moisture",
                    "Apply preventive treatments during high-risk periods"
                ],
                "urgency": "low"
            },
            "potato___early_blight": {
                "description": "Fungal disease with dark spots and target-like patterns",
                "risk_level": "high",
                "actions": [
                    "Apply fungicide spray (chlorothalonil or copper-based)",
                    "Remove infected foliage",
                    "Improve air circulation",
                    "Avoid overhead irrigation"
                ],
                "urgency": "high"
            },
            "potato___late_blight": {
                "description": "Devastating disease that can destroy entire crops",
                "risk_level": "critical",
                "actions": [
                    "IMMEDIATE fungicide application (metalaxyl + mancozeb)",
                    "Destroy infected plants immediately",
                    "Notify neighboring farmers",
                    "Implement strict sanitation measures"
                ],
                "urgency": "critical"
            },
            "tomato___healthy": {
                "description": "Healthy tomato plant with no disease symptoms",
                "risk_level": "low",
                "actions": [
                    "Maintain regular monitoring schedule",
                    "Ensure proper plant spacing",
                    "Continue preventive care practices"
                ],
                "urgency": "low"
            },
            "tomato___bacterial_spot": {
                "description": "Bacterial disease causing small, dark spots on leaves and fruits",
                "risk_level": "high",
                "actions": [
                    "Apply copper-based bactericide",
                    "Remove infected plant material",
                    "Avoid overhead watering",
                    "Use pathogen-free seeds and transplants"
                ],
                "urgency": "high"
            },
            "tomato___early_blight": {
                "description": "Fungal disease with concentric ring patterns on leaves",
                "risk_level": "high",
                "actions": [
                    "Apply fungicide treatment",
                    "Improve air circulation",
                    "Remove lower infected leaves",
                    "Mulch to prevent soil splash"
                ],
                "urgency": "high"
            },
            "tomato___late_blight": {
                "description": "Serious fungal disease affecting leaves and fruits",
                "risk_level": "critical",
                "actions": [
                    "IMMEDIATE fungicide application",
                    "Remove infected plants",
                    "Improve drainage and air flow",
                    "Monitor weather conditions closely"
                ],
                "urgency": "critical"
            },
            "tomato___leaf_mold": {
                "description": "Fungal disease causing yellowing and moldy growth on leaves",
                "risk_level": "medium",
                "actions": [
                    "Improve greenhouse ventilation",
                    "Reduce humidity levels",
                    "Apply appropriate fungicides",
                    "Remove infected leaves"
                ],
                "urgency": "medium"
            },
            "tomato___septoria_leaf_spot": {
                "description": "Fungal disease with small, circular spots on lower leaves",
                "risk_level": "medium",
                "actions": [
                    "Apply fungicide preventively",
                    "Remove infected lower leaves",
                    "Avoid overhead irrigation",
                    "Improve air circulation"
                ],
                "urgency": "medium"
            },
            "tomato___spider_mites_two-spotted_spider_mite": {
                "description": "Pest causing stippling, webbing, and leaf bronzing",
                "risk_level": "medium",
                "actions": [
                    "Apply miticide or insecticidal soap",
                    "Increase humidity levels",
                    "Remove heavily infested leaves",
                    "Introduce beneficial predatory mites"
                ],
                "urgency": "medium"
            },
            "tomato___target_spot": {
                "description": "Fungal disease with target-like spots on leaves",
                "risk_level": "medium",
                "actions": [
                    "Apply fungicide treatment",
                    "Remove infected debris",
                    "Improve air circulation",
                    "Rotate crops"
                ],
                "urgency": "medium"
            },
            "tomato___tomato_mosaic_virus": {
                "description": "Viral disease causing mosaic patterns on leaves",
                "risk_level": "high",
                "actions": [
                    "Remove infected plants immediately",
                    "Control insect vectors",
                    "Use virus-resistant varieties",
                    "Sanitize tools and equipment"
                ],
                "urgency": "high"
            },
            "tomato___tomato_yellow_leaf_curl_virus": {
                "description": "Viral disease transmitted by whiteflies",
                "risk_level": "high",
                "actions": [
                    "Control whitefly populations",
                    "Remove infected plants",
                    "Use reflective mulches",
                    "Plant resistant varieties"
                ],
                "urgency": "high"
            }
        }
    
    def preprocess_image(self, image_input: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
        """Preprocess image for model inference."""
        
        # Load image if path is provided
        if isinstance(image_input, str):
            image = cv2.imread(image_input)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, Image.Image):
            image = np.array(image_input.convert('RGB'))
        else:
            image = image_input.copy()
        
        # Resize to model input size
        image = cv2.resize(image, self.img_size)
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Apply ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image
    
    def predict_single(self, image_input: Union[str, np.ndarray, Image.Image], 
                      return_explanation: bool = True, 
                      mc_samples: int = 10) -> PredictionResult:
        """
        Make prediction on a single image with uncertainty estimation.
        
        Args:
            image_input: Image path, numpy array, or PIL Image
            return_explanation: Whether to generate Grad-CAM explanation
            mc_samples: Number of Monte Carlo samples for uncertainty estimation
            
        Returns:
            PredictionResult with prediction, confidence, and explanations
        """
        
        # Preprocess image
        processed_image = self.preprocess_image(image_input)
        
        # Monte Carlo Dropout for uncertainty estimation
        predictions = []
        for _ in range(mc_samples):
            pred = self.model(processed_image, training=True)  # Enable dropout
            predictions.append(pred.numpy())
        
        # Calculate mean and uncertainty
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        uncertainty = np.std(predictions, axis=0).max()
        
        # Get prediction results
        predicted_class_idx = int(np.argmax(mean_pred[0]))
        confidence = float(mean_pred[0][predicted_class_idx])
        predicted_class_name = self.class_names[predicted_class_idx]
        
        # Create probability dictionary
        all_probabilities = {
            self.class_names[i]: float(mean_pred[0][i]) 
            for i in range(len(self.class_names))
        }
        
        # Generate explanation map if requested
        explanation_map = None
        if return_explanation:
            explanation_map = self._generate_gradcam(processed_image, predicted_class_idx)
        
        # Get recommendations
        recommendations, risk_level = self._get_recommendations(predicted_class_name, confidence)
        
        return PredictionResult(
            class_name=predicted_class_name,
            class_index=predicted_class_idx,
            confidence=confidence,
            all_probabilities=all_probabilities,
            uncertainty=float(uncertainty),
            explanation_map=explanation_map,
            recommendations=recommendations,
            risk_level=risk_level
        )
    
    def _generate_gradcam(self, processed_image: np.ndarray, 
                         predicted_class: int) -> np.ndarray:
        """Generate Grad-CAM visualization for explanation."""
        
        try:
            # Find the last convolutional layer
            last_conv_layer = None
            for layer in reversed(self.model.layers):
                if len(layer.output_shape) == 4:  # Conv layer has 4D output
                    last_conv_layer = layer
                    break
            
            if last_conv_layer is None:
                print("⚠️ No convolutional layer found for Grad-CAM")
                return None
            
            # Create gradient model
            grad_model = tf.keras.Model(
                inputs=self.model.input,
                outputs=[last_conv_layer.output, self.model.output]
            )
            
            # Compute gradients
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(processed_image)
                class_output = predictions[:, predicted_class]
            
            grads = tape.gradient(class_output, conv_outputs)
            if grads is None:
                print("⚠️ Could not compute gradients for Grad-CAM")
                return np.zeros(self.img_size + (1,))
            
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
            # Weight feature maps by gradients
            conv_outputs = conv_outputs[0]
            heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)
            
            # Normalize heatmap
            heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
            
            # Resize to match input image size
            heatmap = tf.image.resize(heatmap[..., tf.newaxis], self.img_size)
            if heatmap is not None:
                heatmap = heatmap.numpy()
            
            return heatmap if heatmap is not None else np.zeros(self.img_size + (1,))
            
        except Exception as e:
            print(f"⚠️ Error generating Grad-CAM: {e}")
            return np.zeros(self.img_size + (1,))
    
    def _get_recommendations(self, class_name: str, confidence: float) -> Tuple[List[str], str]:
        """Get agricultural recommendations based on prediction."""
        
        # Clean class name for lookup
        clean_class_name = class_name.lower().replace(' ', '_')
        
        # Get base recommendations
        if clean_class_name in self.disease_recommendations:
            disease_info = self.disease_recommendations[clean_class_name]
            base_actions = disease_info['actions']
            base_risk = disease_info['urgency']
        else:
            base_actions = ["Consult with agricultural extension service", 
                          "Monitor plant closely", 
                          "Consider laboratory diagnosis"]
            base_risk = "medium"
        
        # Adjust recommendations based on confidence
        recommendations = []
        
        if confidence > 0.9:
            recommendations.extend(base_actions)
        elif confidence > 0.7:
            recommendations.append("Moderate confidence in diagnosis - consider confirming with expert")
            recommendations.extend(base_actions)
        else:
            recommendations.extend([
                "Low confidence prediction - seek expert opinion",
                "Consider additional diagnostic tests",
                "Monitor plant for symptom development"
            ])
            
        # Determine risk level
        if confidence < 0.5:
            risk_level = "uncertain"
        elif "critical" in base_risk and confidence > 0.8:
            risk_level = "critical"
        elif "high" in base_risk and confidence > 0.7:
            risk_level = "high"
        elif "low" in base_risk:
            risk_level = "low"
        else:
            risk_level = "medium"
        
        return recommendations, risk_level
    
    def predict_batch(self, image_paths: List[str], 
                     return_explanations: bool = False) -> List[PredictionResult]:
        """Make predictions on a batch of images."""
        
        results = []
        for image_path in image_paths:
            try:
                result = self.predict_single(image_path, return_explanations)
                results.append(result)
            except Exception as e:
                print(f"❌ Error processing {image_path}: {e}")
                # Create error result
                error_result = PredictionResult(
                    class_name="Error",
                    class_index=-1,
                    confidence=0.0,
                    all_probabilities={},
                    uncertainty=1.0,
                    recommendations=["Unable to process image"],
                    risk_level="unknown"
                )
                results.append(error_result)
        
        return results
    
    def visualize_prediction(self, image_input: Union[str, np.ndarray], 
                           result: PredictionResult, 
                           show_top_k: int = 5) -> go.Figure:
        """Create comprehensive prediction visualization."""
        
        # Load original image for display
        if isinstance(image_input, str):
            original_image = cv2.imread(image_input)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        else:
            original_image = image_input
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Original Image', 'Grad-CAM Explanation',
                'Top Predictions', 'Confidence Analysis'
            ),
            specs=[
                [{"type": "xy"}, {"type": "xy"}],
                [{"type": "xy"}, {"type": "xy"}]
            ]
        )
        
        # Plot original image
        fig.add_trace(
            go.Image(z=original_image),
            row=1, col=1
        )
        
        # Plot Grad-CAM if available
        if result.explanation_map is not None:
            # Overlay heatmap on original image
            heatmap = result.explanation_map[:, :, 0]
            fig.add_trace(
                go.Heatmap(
                    z=heatmap,
                    colorscale='jet',
                    opacity=0.6,
                    showscale=False
                ),
                row=1, col=2
            )
        
        # Top predictions bar chart
        top_probs = dict(sorted(result.all_probabilities.items(), 
                               key=lambda x: x[1], reverse=True)[:show_top_k])
        
        fig.add_trace(
            go.Bar(
                x=list(top_probs.values()),
                y=list(top_probs.keys()),
                orientation='h',
                marker_color='skyblue',
                text=[f'{v:.3f}' for v in top_probs.values()],
                textposition='auto'
            ),
            row=2, col=1
        )
        
        # Confidence gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=result.confidence,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Confidence"},
                gauge={
                    'axis': {'range': [None, 1]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 0.5], 'color': "lightgray"},
                        {'range': [0.5, 0.8], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 0.9
                    }
                }
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text=f"🌱 Plant Disease Prediction: {result.class_name}",
            title_x=0.5,
            showlegend=False
        )
        
        return fig
    
    def generate_report(self, image_path: str, result: PredictionResult) -> Dict[str, Any]:
        """Generate comprehensive diagnostic report."""
        
        # Get disease information
        clean_class_name = result.class_name.lower().replace(' ', '_')
        disease_info = self.disease_recommendations.get(clean_class_name, {})
        
        report = {
            "image_path": str(image_path),
            "timestamp": pd.Timestamp.now().isoformat(),
            "diagnosis": {
                "predicted_disease": result.class_name,
                "confidence_score": f"{result.confidence:.3f}",
                "uncertainty": f"{result.uncertainty:.3f}",
                "risk_level": result.risk_level
            },
            "disease_information": {
                "description": disease_info.get("description", "No description available"),
                "urgency": disease_info.get("urgency", "unknown")
            },
            "recommendations": {
                "immediate_actions": result.recommendations[:3] if result.recommendations else [],
                "monitoring_advice": result.recommendations[3:] if result.recommendations and len(result.recommendations) > 3 else [],
                "follow_up": ["Recheck plant condition in 3-5 days", 
                             "Document treatment effectiveness"]
            },
            "top_predictions": dict(list(result.all_probabilities.items())[:5]),
            "model_info": {
                "model_path": str(self.model_path),
                "input_size": self.img_size,
                "num_classes": self.num_classes
            }
        }
        
        return report

# Utility functions for batch processing
def process_directory(inference_engine: PlantDiseaseInference, 
                     image_directory: str, 
                     output_path: Optional[str] = None) -> pd.DataFrame:
    """Process all images in a directory and create results DataFrame."""
    
    image_dir = Path(image_directory)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    # Find all image files
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(image_dir.glob(f'*{ext}'))
        image_paths.extend(image_dir.glob(f'*{ext.upper()}'))
    
    print(f"🔍 Found {len(image_paths)} images to process")
    
    # Process images
    results = inference_engine.predict_batch([str(p) for p in image_paths])
    
    # Create DataFrame
    data = []
    for path, result in zip(image_paths, results):
        data.append({
            'image_path': str(path),
            'filename': path.name,
            'predicted_class': result.class_name,
            'confidence': result.confidence,
            'uncertainty': result.uncertainty,
            'risk_level': result.risk_level,
            'top_prediction_1': max(result.all_probabilities.items(), key=lambda x: x[1])[0],
            'top_confidence_1': max(result.all_probabilities.values()),
            'recommendations_count': len(result.recommendations) if result.recommendations else 0
        })
    
    df = pd.DataFrame(data)
    
    # Save results if output path provided
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"💾 Results saved to: {output_path}")
    
    return df

if __name__ == "__main__":
    # Example usage
    print("🧠 CAPSTONE-LAZARUS Inference Engine")
    print("=" * 50)
    
    # This would be used after model training
    # model_path = "../models/efficientnet_v2_b0_plant_disease_best.h5"
    # class_names = [...] # Load from saved class mapping
    # 
    # inference_engine = PlantDiseaseInference(model_path, class_names)
    # result = inference_engine.predict_single("path/to/test/image.jpg")
    # print(f"Prediction: {result.class_name} (confidence: {result.confidence:.3f})")
    