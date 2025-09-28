"""
Explainability Studio Route - Lazarus Console  
Grad-CAM, SHAP, and model interpretation tools
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter
import io
import base64
from components.state_manager import get_state, set_state, add_decision_log_entry
from utils.model_manager import ModelManager

def render_explainability_studio():
    """Render explainability studio interface"""
    
    st.markdown("## 🎯 Explainability Studio")
    st.markdown("*Model interpretation and visualization workspace*")
    
    # Initialize model manager
    if 'model_manager' not in st.session_state:
        st.session_state.model_manager = ModelManager()
    
    # Explainability controls
    render_explainability_controls()
    
    # Main workspace
    tab1, tab2, tab3, tab4 = st.tabs(["🔥 Grad-CAM", "🧩 SHAP Analysis", "🔍 Feature Maps", "📊 Model Insights"])
    
    with tab1:
        render_gradcam_interface()
    
    with tab2:
        render_shap_interface()
    
    with tab3:
        render_feature_maps()
    
    with tab4:
        render_model_insights()

def render_explainability_controls():
    """Render explainability control panel"""
    
    st.markdown("### Explanation Configuration")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        model_options = ["EfficientNet-B0", "ResNet-50", "MobileNet-V2", "Vision Transformer"]
        selected_model = st.selectbox("Model", model_options, index=0)
        set_state('explain_model', selected_model)
    
    with col2:
        layer_options = ["Last Conv", "Feature Layer", "Attention", "Custom"]
        target_layer = st.selectbox("Target Layer", layer_options, index=0)
        set_state('target_layer', target_layer)
    
    with col3:
        class_options = ["Predicted Class", "Specific Class", "All Classes"]
        target_class = st.selectbox("Target Class", class_options, index=0)
        set_state('target_class', target_class)
    
    with col4:
        colormap_options = ["jet", "viridis", "plasma", "inferno", "coolwarm"]
        colormap = st.selectbox("Colormap", colormap_options, index=0)
        set_state('explanation_colormap', colormap)

def render_gradcam_interface():
    """Render Grad-CAM visualization interface"""
    
    st.markdown("### Grad-CAM Heatmaps")
    st.markdown("*Gradient-weighted Class Activation Mapping*")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Input Image")
        
        uploaded_file = st.file_uploader("Choose an image", type=['png', 'jpg', 'jpeg'], key="gradcam_upload")
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Original Image", use_column_width=True)
            
            # Grad-CAM options
            st.markdown("#### Grad-CAM Options")
            
            alpha = st.slider("Overlay Alpha", 0.0, 1.0, 0.6, 0.1)
            guided_backprop = st.checkbox("Guided Backpropagation", value=False)
            
            if st.button("🔥 Generate Grad-CAM", type="primary"):
                with st.spinner("Generating heatmap..."):
                    gradcam_result = generate_mock_gradcam(image)
                    set_state('gradcam_result', gradcam_result)
                    st.success("Grad-CAM generated!")
    
    with col2:
        st.markdown("#### Grad-CAM Visualization")
        
        gradcam_result = get_state('gradcam_result')
        if gradcam_result:
            render_gradcam_results(gradcam_result)

def render_shap_interface():
    """Render SHAP analysis interface"""
    
    st.markdown("### SHAP (SHapley Additive exPlanations)")
    st.markdown("*Feature importance and attribution analysis*")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### SHAP Configuration")
        
        shap_type = st.selectbox("SHAP Explainer", [
            "Deep Explainer",
            "Gradient Explainer", 
            "Partition Explainer",
            "Kernel Explainer"
        ])
        
        num_samples = st.number_input("Background Samples", min_value=10, max_value=1000, value=100)
        
        uploaded_file = st.file_uploader("Choose an image", type=['png', 'jpg', 'jpeg'], key="shap_upload")
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Input Image", use_column_width=True)
            
            if st.button("🧩 Generate SHAP", type="primary"):
                with st.spinner("Computing SHAP values..."):
                    shap_result = generate_mock_shap(image)
                    set_state('shap_result', shap_result)
                    st.success("SHAP analysis completed!")
    
    with col2:
        st.markdown("#### SHAP Analysis")
        
        shap_result = get_state('shap_result')
        if shap_result:
            render_shap_results(shap_result)

def render_feature_maps():
    """Render feature map visualization"""
    
    st.markdown("### Feature Map Visualization")
    st.markdown("*Intermediate layer activations*")
    
    # Layer selection
    col1, col2, col3 = st.columns(3)
    
    with col1:
        layer_depth = st.selectbox("Layer Depth", ["Shallow", "Middle", "Deep"], index=1)
    
    with col2:
        num_filters = st.number_input("Filters to Show", min_value=1, max_value=64, value=16)
    
    with col3:
        normalization = st.selectbox("Normalization", ["MinMax", "Z-Score", "None"], index=0)
    
    # Image upload
    uploaded_file = st.file_uploader("Choose an image", type=['png', 'jpg', 'jpeg'], key="feature_upload")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image(image, caption="Input Image", use_column_width=True)
            
            if st.button("🔍 Extract Features", type="primary"):
                with st.spinner("Extracting feature maps..."):
                    feature_maps = generate_mock_feature_maps(num_filters)
                    set_state('feature_maps', feature_maps)
                    st.success("Feature maps extracted!")
        
        with col2:
            feature_maps = get_state('feature_maps')
            if feature_maps:
                render_feature_map_grid(feature_maps, num_filters)

def render_model_insights():
    """Render model insights and statistics"""
    
    st.markdown("### Model Insights")
    st.markdown("*Global model behavior and statistics*")
    
    # Model statistics
    render_model_statistics()
    
    # Class activation patterns
    render_class_patterns()
    
    # Decision boundaries
    render_decision_boundaries()

def render_gradcam_results(gradcam_result):
    """Render Grad-CAM visualization results"""
    
    # Prediction info
    st.info(f"**Predicted:** {gradcam_result['prediction']} ({gradcam_result['confidence']:.1%})")
    
    # Grad-CAM heatmap
    st.image(gradcam_result['heatmap'], caption="Grad-CAM Heatmap", use_column_width=True)
    
    # Overlay
    st.image(gradcam_result['overlay'], caption="Overlay", use_column_width=True)
    
    # Attribution scores
    st.markdown("#### Attribution Analysis")
    
    scores_df = pd.DataFrame(gradcam_result['attribution_scores'])
    fig = px.bar(
        scores_df, 
        x='region', 
        y='score',
        title="Region Attribution Scores",
        color='score',
        color_continuous_scale='viridis'
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)

def render_shap_results(shap_result):
    """Render SHAP analysis results"""
    
    # SHAP values heatmap
    st.image(shap_result['shap_heatmap'], caption="SHAP Attribution", use_column_width=True)
    
    # Feature importance
    st.markdown("#### Feature Importance")
    
    importance_df = pd.DataFrame(shap_result['feature_importance'])
    fig = px.bar(
        importance_df,
        x='importance',
        y='feature',
        orientation='h',
        title="SHAP Feature Importance",
        color='importance',
        color_continuous_scale='RdBu'
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # SHAP waterfall
    st.markdown("#### SHAP Waterfall")
    
    waterfall_data = shap_result['waterfall_data']
    fig = go.Figure(go.Waterfall(
        name="SHAP",
        orientation="v",
        measure=["relative", "relative", "relative", "relative", "total"],
        x=waterfall_data['features'],
        textposition="outside",
        text=waterfall_data['values'],
        y=waterfall_data['values'],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
    ))
    
    fig.update_layout(
        title="SHAP Value Contributions",
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)

def render_feature_map_grid(feature_maps, num_filters):
    """Render feature map grid visualization"""
    
    st.markdown("#### Feature Maps Grid")
    
    # Calculate grid dimensions
    cols = min(4, num_filters)
    rows = (num_filters + cols - 1) // cols
    
    # Create grid of feature maps
    for row in range(rows):
        columns = st.columns(cols)
        for col in range(cols):
            idx = row * cols + col
            if idx < len(feature_maps):
                with columns[col]:
                    st.image(
                        feature_maps[idx], 
                        caption=f"Filter {idx+1}",
                        use_column_width=True
                    )

def render_model_statistics():
    """Render global model statistics"""
    
    st.markdown("#### Model Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Parameters", "23.5M")
    
    with col2:
        st.metric("Trainable Params", "23.2M")
    
    with col3:
        st.metric("Model Size", "94.2 MB")
    
    with col4:
        st.metric("FLOPS", "4.2B")
    
    # Layer statistics
    st.markdown("#### Layer Analysis")
    
    layer_stats = pd.DataFrame({
        'Layer': ['Conv1', 'Conv2', 'Conv3', 'Conv4', 'Conv5', 'FC'],
        'Parameters': [9408, 147456, 294912, 589824, 1179648, 20971520],
        'Output Shape': ['(64, 224, 224)', '(128, 112, 112)', '(256, 56, 56)', 
                        '(512, 28, 28)', '(1024, 14, 14)', '(10,)'],
        'Activation': ['ReLU', 'ReLU', 'ReLU', 'ReLU', 'ReLU', 'Softmax']
    })
    
    st.dataframe(layer_stats, use_container_width=True)

def render_class_patterns():
    """Render class activation patterns"""
    
    st.markdown("#### Class Activation Patterns")
    
    # Mock class patterns data
    classes = ['Healthy', 'Blight', 'Rust', 'Spot', 'Mosaic']
    patterns_data = []
    
    for cls in classes:
        for region in ['Center', 'Edges', 'Corners', 'Background']:
            activation = np.random.uniform(0.1, 0.9)
            patterns_data.append({
                'Class': cls,
                'Region': region,
                'Activation': activation
            })
    
    patterns_df = pd.DataFrame(patterns_data)
    
    fig = px.imshow(
        patterns_df.pivot(index='Class', columns='Region', values='Activation'),
        color_continuous_scale='viridis',
        title="Class-Region Activation Patterns"
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)

def render_decision_boundaries():
    """Render decision boundary visualization"""
    
    st.markdown("#### Decision Boundaries (t-SNE)")
    
    # Mock t-SNE data
    np.random.seed(42)
    n_samples = 500
    
    tsne_data = []
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    classes = ['Healthy', 'Blight', 'Rust', 'Spot', 'Mosaic']
    
    for i, (cls, color) in enumerate(zip(classes, colors)):
        x = np.random.normal(i*3, 1.5, n_samples//5)
        y = np.random.normal(i*2, 1.2, n_samples//5)
        
        for j in range(n_samples//5):
            tsne_data.append({
                'x': x[j],
                'y': y[j],
                'class': cls,
                'color': color
            })
    
    tsne_df = pd.DataFrame(tsne_data)
    
    fig = px.scatter(
        tsne_df, 
        x='x', 
        y='y', 
        color='class',
        title="t-SNE Feature Space Visualization",
        opacity=0.7
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis_title="t-SNE Component 1",
        yaxis_title="t-SNE Component 2"
    )
    st.plotly_chart(fig, use_container_width=True)

def generate_mock_gradcam(image):
    """Generate mock Grad-CAM results"""
    
    # Create mock heatmap
    width, height = image.size
    heatmap_array = np.random.rand(height//4, width//4)  # Smaller resolution
    heatmap_array = (heatmap_array * 255).astype(np.uint8)
    
    # Convert to PIL and resize
    heatmap = Image.fromarray(heatmap_array, mode='L')
    heatmap = heatmap.resize((width, height))
    
    # Create colored heatmap
    heatmap_colored = Image.new('RGB', (width, height))
    pixels = heatmap.load()
    colored_pixels = heatmap_colored.load()
    
    for i in range(width):
        for j in range(height):
            intensity = pixels[i, j]
            # Jet colormap approximation
            if intensity < 64:
                colored_pixels[i, j] = (0, 0, intensity*4)
            elif intensity < 128:
                colored_pixels[i, j] = (0, (intensity-64)*4, 255)
            elif intensity < 192:
                colored_pixels[i, j] = ((intensity-128)*4, 255, 255-(intensity-128)*4)
            else:
                colored_pixels[i, j] = (255, 255-(intensity-192)*4, 0)
    
    # Create overlay
    overlay = Image.blend(image, heatmap_colored, 0.4)
    
    # Mock attribution scores
    attribution_scores = [
        {'region': 'Center', 'score': 0.75},
        {'region': 'Top-Left', 'score': 0.23},
        {'region': 'Top-Right', 'score': 0.45},
        {'region': 'Bottom-Left', 'score': 0.12},
        {'region': 'Bottom-Right', 'score': 0.38}
    ]
    
    return {
        'prediction': 'Blight',
        'confidence': 0.89,
        'heatmap': heatmap_colored,
        'overlay': overlay,
        'attribution_scores': attribution_scores
    }

def generate_mock_shap(image):
    """Generate mock SHAP results"""
    
    # Mock SHAP heatmap (similar to Grad-CAM)
    width, height = image.size
    shap_array = np.random.randn(height//4, width//4)  # Can be negative
    shap_array = ((shap_array + 2) * 63.75).astype(np.uint8)  # Scale to 0-255
    
    shap_heatmap = Image.fromarray(shap_array, mode='L')
    shap_heatmap = shap_heatmap.resize((width, height))
    
    # Mock feature importance
    feature_importance = [
        {'feature': 'Leaf Texture', 'importance': 0.34},
        {'feature': 'Color Pattern', 'importance': 0.28},
        {'feature': 'Edge Sharpness', 'importance': 0.19},
        {'feature': 'Brightness', 'importance': 0.12},
        {'feature': 'Contrast', 'importance': 0.07}
    ]
    
    # Mock waterfall data
    waterfall_data = {
        'features': ['Base Rate', 'Texture', 'Color', 'Edges', 'Final'],
        'values': [0.2, 0.15, 0.08, -0.05, 0.38]
    }
    
    return {
        'shap_heatmap': shap_heatmap,
        'feature_importance': feature_importance,
        'waterfall_data': waterfall_data
    }

def generate_mock_feature_maps(num_filters):
    """Generate mock feature maps"""
    
    feature_maps = []
    
    for i in range(num_filters):
        # Generate random feature map
        feature_map = np.random.rand(64, 64) * 255
        feature_map = feature_map.astype(np.uint8)
        
        # Convert to PIL Image
        feature_img = Image.fromarray(feature_map, mode='L')
        feature_maps.append(feature_img)
    
    return feature_maps