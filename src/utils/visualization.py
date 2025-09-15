"""
Visualization utilities for CAPSTONE-LAZARUS
===========================================
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import cv2
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def plot_training_history(history: tf.keras.callbacks.History,
                         save_path: Optional[str] = None,
                         show_plot: bool = True) -> go.Figure:
    """
    Plot training history with interactive Plotly
    
    Args:
        history: Training history
        save_path: Optional path to save plot
        show_plot: Whether to display plot
        
    Returns:
        Plotly figure
    """
    
    # Extract metrics
    metrics = list(history.history.keys())
    epochs = range(1, len(history.history[metrics[0]]) + 1)
    
    # Separate training and validation metrics
    train_metrics = [m for m in metrics if not m.startswith('val_')]
    val_metrics = [m for m in metrics if m.startswith('val_')]
    
    # Create subplots
    n_metrics = len(train_metrics)
    fig = make_subplots(
        rows=n_metrics, cols=1,
        subplot_titles=[m.replace('_', ' ').title() for m in train_metrics],
        vertical_spacing=0.08
    )
    
    colors = px.colors.qualitative.Set1
    
    for i, metric in enumerate(train_metrics):
        # Training metric
        fig.add_trace(
            go.Scatter(
                x=list(epochs),
                y=history.history[metric],
                mode='lines+markers',
                name=f'Training {metric}',
                line=dict(color=colors[i % len(colors)]),
                marker=dict(size=4)
            ),
            row=i+1, col=1
        )
        
        # Validation metric if available
        val_metric = f'val_{metric}'
        if val_metric in val_metrics:
            fig.add_trace(
                go.Scatter(
                    x=list(epochs),
                    y=history.history[val_metric],
                    mode='lines+markers',
                    name=f'Validation {metric}',
                    line=dict(color=colors[i % len(colors)], dash='dash'),
                    marker=dict(size=4)
                ),
                row=i+1, col=1
            )
    
    # Update layout
    fig.update_layout(
        title="Training History",
        height=300 * n_metrics,
        showlegend=True,
        template="plotly_white"
    )
    
    fig.update_xaxes(title="Epoch")
    
    if save_path:
        fig.write_html(str(save_path))
    
    if show_plot:
        fig.show()
    
    return fig


def create_confusion_matrix(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          class_names: Optional[List[str]] = None,
                          normalize: str = 'true',
                          save_path: Optional[str] = None) -> go.Figure:
    """
    Create interactive confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Class names
        normalize: Normalization method ('true', 'pred', 'all', None)
        save_path: Optional path to save plot
        
    Returns:
        Plotly figure
    """
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, normalize=normalize)
    
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(cm))]
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=class_names,
        y=class_names,
        colorscale='Blues',
        text=np.round(cm, 3),
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title=f'Confusion Matrix (normalize={normalize})',
        xaxis_title='Predicted Label',
        yaxis_title='True Label',
        width=600,
        height=600
    )
    
    if save_path:
        fig.write_html(str(save_path))
    
    return fig


def plot_class_distribution(labels: List[str],
                           class_names: Optional[List[str]] = None,
                           save_path: Optional[str] = None) -> go.Figure:
    """
    Plot class distribution
    
    Args:
        labels: List of class labels
        class_names: Class names
        save_path: Optional path to save plot
        
    Returns:
        Plotly figure
    """
    
    # Count classes
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    if class_names is None:
        class_names = [f'Class {i}' for i in unique_labels]
    else:
        class_names = [class_names[i] for i in unique_labels]
    
    # Create bar plot
    fig = go.Figure(data=[
        go.Bar(
            x=class_names,
            y=counts,
            marker_color='skyblue',
            text=counts,
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title='Class Distribution',
        xaxis_title='Classes',
        yaxis_title='Count',
        template="plotly_white"
    )
    
    fig.update_xaxes(tickangle=45)
    
    if save_path:
        fig.write_html(str(save_path))
    
    return fig


def plot_model_comparison(models_performance: Dict[str, Dict[str, float]],
                         save_path: Optional[str] = None) -> go.Figure:
    """
    Compare model performance
    
    Args:
        models_performance: Dictionary of model names and their metrics
        save_path: Optional path to save plot
        
    Returns:
        Plotly figure
    """
    
    # Extract data
    model_names = list(models_performance.keys())
    metrics = list(next(iter(models_performance.values())).keys())
    
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set1
    
    for i, metric in enumerate(metrics):
        values = [models_performance[model][metric] for model in model_names]
        
        fig.add_trace(go.Bar(
            name=metric.replace('_', ' ').title(),
            x=model_names,
            y=values,
            marker_color=colors[i % len(colors)],
            text=[f'{v:.3f}' for v in values],
            textposition='auto'
        ))
    
    fig.update_layout(
        title='Model Performance Comparison',
        xaxis_title='Models',
        yaxis_title='Performance',
        barmode='group',
        template="plotly_white"
    )
    
    if save_path:
        fig.write_html(str(save_path))
    
    return fig


def plot_feature_embeddings(features: np.ndarray,
                           labels: np.ndarray,
                           method: str = 'tsne',
                           class_names: Optional[List[str]] = None,
                           save_path: Optional[str] = None) -> go.Figure:
    """
    Visualize feature embeddings using dimensionality reduction
    
    Args:
        features: Feature vectors
        labels: Class labels
        method: Reduction method ('tsne' or 'pca')
        class_names: Class names
        save_path: Optional path to save plot
        
    Returns:
        Plotly figure
    """
    
    # Reduce dimensions
    if method.lower() == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)-1))
    elif method.lower() == 'pca':
        reducer = PCA(n_components=2, random_state=42)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    reduced_features = reducer.fit_transform(features)
    
    # Create scatter plot
    if class_names is None:
        class_names = [f'Class {i}' for i in range(max(labels) + 1)]
    
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set1
    
    for class_idx in np.unique(labels):
        mask = labels == class_idx
        
        fig.add_trace(go.Scatter(
            x=reduced_features[mask, 0],
            y=reduced_features[mask, 1],
            mode='markers',
            name=class_names[class_idx] if class_idx < len(class_names) else f'Class {class_idx}',
            marker=dict(
                color=colors[class_idx % len(colors)],
                size=8,
                opacity=0.7
            ),
            hovertemplate=f'<b>{class_names[class_idx] if class_idx < len(class_names) else f"Class {class_idx}"}</b><br>' +
                         'X: %{x:.2f}<br>' +
                         'Y: %{y:.2f}<extra></extra>'
        ))
    
    fig.update_layout(
        title=f'Feature Embeddings ({method.upper()})',
        xaxis_title=f'{method.upper()}-1',
        yaxis_title=f'{method.upper()}-2',
        template="plotly_white",
        hovermode='closest'
    )
    
    if save_path:
        fig.write_html(str(save_path))
    
    return fig


def plot_prediction_confidence(predictions: List[float],
                              threshold: float = 0.5,
                              save_path: Optional[str] = None) -> go.Figure:
    """
    Plot prediction confidence distribution
    
    Args:
        predictions: List of prediction confidences
        threshold: Confidence threshold
        save_path: Optional path to save plot
        
    Returns:
        Plotly figure
    """
    
    fig = go.Figure()
    
    # Histogram of confidences
    fig.add_trace(go.Histogram(
        x=predictions,
        nbinsx=30,
        name='Confidence Distribution',
        marker_color='skyblue',
        opacity=0.7
    ))
    
    # Add threshold line
    fig.add_vline(
        x=threshold,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Threshold: {threshold}"
    )
    
    fig.update_layout(
        title='Prediction Confidence Distribution',
        xaxis_title='Confidence',
        yaxis_title='Frequency',
        template="plotly_white"
    )
    
    if save_path:
        fig.write_html(str(save_path))
    
    return fig


def visualize_gradcam(model: tf.keras.Model,
                     image: np.ndarray,
                     class_index: int,
                     layer_name: Optional[str] = None,
                     save_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate Grad-CAM visualization
    
    Args:
        model: Trained model
        image: Input image
        class_index: Target class index
        layer_name: Layer name for visualization (last conv layer if None)
        save_path: Optional path to save visualization
        
    Returns:
        Tuple of (heatmap, superimposed_image)
    """
    
    # Find the last convolutional layer if not specified
    if layer_name is None:
        for layer in reversed(model.layers):
            if 'conv' in layer.name.lower():
                layer_name = layer.name
                break
    
    if layer_name is None:
        raise ValueError("No convolutional layer found in model")
    
    # Create gradient model
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(layer_name).output, model.output]
    )
    
    # Calculate gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(tf.expand_dims(image, 0))
        loss = predictions[:, class_index]
    
    # Calculate gradients
    grads = tape.gradient(loss, conv_outputs)
    
    # Global average pooling on gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight the feature maps
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Normalize heatmap
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    # Resize heatmap to match image size
    heatmap_resized = cv2.resize(heatmap.numpy(), (image.shape[1], image.shape[0]))
    
    # Convert to uint8
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    
    # Apply colormap
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    
    # Superimpose on original image
    image_uint8 = np.uint8(255 * image) if image.max() <= 1.0 else np.uint8(image)
    superimposed = 0.6 * image_uint8 + 0.4 * heatmap_colored
    
    if save_path:
        # Save visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(image_uint8)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(heatmap_resized, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap')
        axes[1].axis('off')
        
        axes[2].imshow(np.uint8(superimposed))
        axes[2].set_title('Superimposed')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    return heatmap_resized, np.uint8(superimposed)


def plot_learning_curves(train_sizes: np.ndarray,
                        train_scores: np.ndarray,
                        val_scores: np.ndarray,
                        save_path: Optional[str] = None) -> go.Figure:
    """
    Plot learning curves
    
    Args:
        train_sizes: Training set sizes
        train_scores: Training scores
        val_scores: Validation scores
        save_path: Optional path to save plot
        
    Returns:
        Plotly figure
    """
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    fig = go.Figure()
    
    # Training curve
    fig.add_trace(go.Scatter(
        x=train_sizes,
        y=train_mean + train_std,
        mode='lines',
        line=dict(color='rgba(0,100,80,0)'),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter(
        x=train_sizes,
        y=train_mean - train_std,
        mode='lines',
        line=dict(color='rgba(0,100,80,0)'),
        fillcolor='rgba(0,100,80,0.2)',
        fill='tonexty',
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter(
        x=train_sizes,
        y=train_mean,
        mode='lines+markers',
        name='Training Score',
        line=dict(color='rgb(0,100,80)'),
        marker=dict(size=6)
    ))
    
    # Validation curve
    fig.add_trace(go.Scatter(
        x=train_sizes,
        y=val_mean + val_std,
        mode='lines',
        line=dict(color='rgba(255,0,0,0)'),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter(
        x=train_sizes,
        y=val_mean - val_std,
        mode='lines',
        line=dict(color='rgba(255,0,0,0)'),
        fillcolor='rgba(255,0,0,0.2)',
        fill='tonexty',
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter(
        x=train_sizes,
        y=val_mean,
        mode='lines+markers',
        name='Validation Score',
        line=dict(color='rgb(255,0,0)'),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title='Learning Curves',
        xaxis_title='Training Set Size',
        yaxis_title='Score',
        template="plotly_white",
        hovermode='x unified'
    )
    
    if save_path:
        fig.write_html(str(save_path))
    
    return fig


def create_model_architecture_plot(model: tf.keras.Model,
                                 save_path: Optional[str] = None) -> go.Figure:
    """
    Create model architecture visualization
    
    Args:
        model: Keras model
        save_path: Optional path to save plot
        
    Returns:
        Plotly figure
    """
    
    # Extract layer information
    layers_info = []
    
    for i, layer in enumerate(model.layers):
        layer_info = {
            'layer_index': i,
            'layer_name': layer.name,
            'layer_type': type(layer).__name__,
            'output_shape': str(layer.output_shape),
            'params': layer.count_params() if hasattr(layer, 'count_params') else 0,
            'trainable': layer.trainable
        }
        layers_info.append(layer_info)
    
    df = pd.DataFrame(layers_info)
    
    # Create visualization
    fig = go.Figure()
    
    # Bar plot of parameters per layer
    fig.add_trace(go.Bar(
        x=df['layer_index'],
        y=df['params'],
        text=df['layer_type'],
        textposition='auto',
        name='Parameters',
        hovertemplate='<b>%{text}</b><br>' +
                     'Layer: %{x}<br>' +
                     'Parameters: %{y:,}<br>' +
                     '<extra></extra>'
    ))
    
    fig.update_layout(
        title='Model Architecture - Parameters per Layer',
        xaxis_title='Layer Index',
        yaxis_title='Number of Parameters',
        template="plotly_white",
        showlegend=False
    )
    
    if save_path:
        fig.write_html(str(save_path))
    
    return fig


# Utility functions for batch processing
def save_multiple_plots(figures: Dict[str, go.Figure],
                       output_dir: str,
                       format: str = 'html') -> bool:
    """
    Save multiple plots to directory
    
    Args:
        figures: Dictionary of figure names and plotly figures
        output_dir: Output directory
        format: Output format ('html', 'png', 'pdf')
        
    Returns:
        Success status
    """
    
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for name, fig in figures.items():
            file_path = output_path / f"{name}.{format}"
            
            if format == 'html':
                fig.write_html(str(file_path))
            elif format == 'png':
                fig.write_image(str(file_path))
            elif format == 'pdf':
                fig.write_image(str(file_path))
            else:
                raise ValueError(f"Unsupported format: {format}")
        
        return True
        
    except Exception as e:
        print(f"Error saving plots: {e}")
        return False


def create_dashboard_layout(figures: List[go.Figure],
                           titles: List[str],
                           save_path: Optional[str] = None) -> str:
    """
    Create HTML dashboard with multiple plots
    
    Args:
        figures: List of plotly figures
        titles: List of plot titles
        save_path: Optional path to save dashboard
        
    Returns:
        HTML content
    """
    
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>ML Dashboard</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .plot-container { margin: 20px 0; }
            h1, h2 { color: #333; }
        </style>
    </head>
    <body>
        <h1>🌱 CAPSTONE-LAZARUS Dashboard</h1>
    """
    
    for i, (fig, title) in enumerate(zip(figures, titles)):
        html_content += f"""
        <div class="plot-container">
            <h2>{title}</h2>
            <div id="plot{i}"></div>
            <script>
                Plotly.newPlot('plot{i}', {fig.to_json()});
            </script>
        </div>
        """
    
    html_content += """
    </body>
    </html>
    """
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(html_content)
    
    return html_content