"""
Inference Lab Route - Lazarus Console
Batch inference, model comparison, and result visualization
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from PIL import Image
import io
import base64
from components.state_manager import get_state, set_state, add_decision_log_entry
from utils.model_manager import ModelManager
from utils.dataset_manager import DatasetManager

def render_inference_lab():
    """Render inference lab interface"""
    
    st.markdown("## 🔬 Inference Lab")
    st.markdown("*Batch inference and model comparison workspace*")
    
    # Initialize managers
    if 'model_manager' not in st.session_state:
        st.session_state.model_manager = ModelManager()
    if 'dataset_manager' not in st.session_state:
        st.session_state.dataset_manager = DatasetManager()
    
    # Inference controls
    render_inference_controls()
    
    # Main workspace
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Single Inference", "📊 Batch Processing", "🔍 Model Comparison", "📈 Results Analysis"])
    
    with tab1:
        render_single_inference()
    
    with tab2:
        render_batch_processing()
    
    with tab3:
        render_model_comparison()
    
    with tab4:
        render_results_analysis()

def render_inference_controls():
    """Render inference control panel"""
    
    st.markdown("### Inference Configuration")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        model_options = ["EfficientNet-B0", "ResNet-50", "MobileNet-V2", "Vision Transformer"]
        selected_model = st.selectbox("Model", model_options, index=0)
        set_state('inference_model', selected_model)
    
    with col2:
        format_options = ["PyTorch", "ONNX", "TensorRT"]
        model_format = st.selectbox("Format", format_options, index=0)
        set_state('inference_format', model_format)
    
    with col3:
        batch_size = st.number_input("Batch Size", min_value=1, max_value=64, value=8)
        set_state('inference_batch_size', batch_size)
    
    with col4:
        confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.7, 0.01)
        set_state('confidence_threshold', confidence_threshold)

def render_single_inference():
    """Render single image inference interface"""
    
    st.markdown("### Single Image Inference")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Choose an image", type=['png', 'jpg', 'jpeg'])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Input Image", use_column_width=True)
            
            if st.button("🔍 Run Inference", type="primary"):
                with st.spinner("Processing..."):
                    # Simulate inference
                    results = simulate_inference(image)
                    set_state('single_inference_results', results)
                    st.success("Inference completed!")
    
    with col2:
        results = get_state('single_inference_results')
        if results:
            render_inference_results(results)

def render_batch_processing():
    """Render batch processing interface"""
    
    st.markdown("### Batch Processing")
    
    # Data source selection
    col1, col2 = st.columns(2)
    
    with col1:
        data_source = st.radio("Data Source", ["Upload Folder", "Dataset Directory", "From Manifest"])
        
        if data_source == "Upload Folder":
            uploaded_files = st.file_uploader("Choose images", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)
            if uploaded_files:
                st.info(f"Selected {len(uploaded_files)} images")
        
        elif data_source == "Dataset Directory":
            dataset_path = st.text_input("Dataset Path", value="data/")
        
        else:  # From Manifest
            manifest_file = st.file_uploader("Upload Manifest", type=['csv'])
    
    with col2:
        # Processing options
        st.markdown("#### Processing Options")
        save_predictions = st.checkbox("Save Predictions", value=True)
        generate_report = st.checkbox("Generate Report", value=True)
        export_format = st.selectbox("Export Format", ["CSV", "JSON", "Excel"])
    
    # Batch processing controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🚀 Start Batch Processing", type="primary"):
            batch_results = simulate_batch_processing()
            set_state('batch_results', batch_results)
            st.success("Batch processing completed!")
    
    with col2:
        processing_status = get_state('batch_processing_status', 'idle')
        st.metric("Status", processing_status.upper())
    
    with col3:
        if get_state('batch_results'):
            if st.button("📥 Download Results"):
                st.success("Results downloaded!")
    
    # Progress and results
    batch_results = get_state('batch_results')
    if batch_results:
        render_batch_results(batch_results)

def render_model_comparison():
    """Render model comparison interface"""
    
    st.markdown("### Model Comparison")
    
    # Model selection for comparison
    st.markdown("#### Select Models to Compare")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        model1 = st.selectbox("Model 1", ["EfficientNet-B0", "ResNet-50", "MobileNet-V2"], key="comp_model1")
    
    with col2:
        model2 = st.selectbox("Model 2", ["EfficientNet-B0", "ResNet-50", "MobileNet-V2"], key="comp_model2")
    
    with col3:
        test_dataset = st.selectbox("Test Dataset", ["Validation Set", "Test Set", "Custom Images"])
    
    # Comparison metrics
    if st.button("🔄 Run Comparison", type="primary"):
        comparison_results = simulate_model_comparison(model1, model2)
        set_state('comparison_results', comparison_results)
        st.success("Comparison completed!")
    
    # Display comparison results
    comparison_results = get_state('comparison_results')
    if comparison_results:
        render_comparison_results(comparison_results)

def render_results_analysis():
    """Render results analysis interface"""
    
    st.markdown("### Results Analysis")
    
    # Analysis options
    col1, col2 = st.columns(2)
    
    with col1:
        analysis_type = st.selectbox("Analysis Type", [
            "Prediction Confidence Distribution",
            "Class-wise Performance",
            "Error Analysis",
            "Prediction Patterns"
        ])
    
    with col2:
        visualization_type = st.selectbox("Visualization", [
            "Histogram",
            "Box Plot",
            "Confusion Matrix",
            "ROC Curves"
        ])
    
    # Generate analysis
    if st.button("📊 Generate Analysis"):
        analysis_data = generate_analysis_data(analysis_type)
        render_analysis_visualization(analysis_data, analysis_type, visualization_type)

def render_inference_results(results):
    """Render single inference results"""
    
    st.markdown("### Prediction Results")
    
    # Top prediction
    top_class = results['predictions'][0]
    st.success(f"**Predicted Class:** {top_class['class']}")
    st.success(f"**Confidence:** {top_class['confidence']:.2%}")
    
    # All predictions
    st.markdown("#### All Predictions")
    predictions_df = pd.DataFrame(results['predictions'])
    
    # Create horizontal bar chart
    fig = px.bar(
        predictions_df, 
        x='confidence', 
        y='class', 
        orientation='h',
        color='confidence',
        color_continuous_scale='viridis'
    )
    fig.update_layout(
        height=300,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)

def render_batch_results(batch_results):
    """Render batch processing results"""
    
    st.markdown("### Batch Processing Results")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Images", batch_results['total_images'])
    
    with col2:
        st.metric("Processed", batch_results['processed'])
    
    with col3:
        st.metric("Avg Confidence", f"{batch_results['avg_confidence']:.2%}")
    
    with col4:
        st.metric("Processing Time", f"{batch_results['processing_time']:.1f}s")
    
    # Class distribution
    st.markdown("#### Predicted Class Distribution")
    class_counts = batch_results['class_distribution']
    
    fig = px.pie(
        values=list(class_counts.values()),
        names=list(class_counts.keys()),
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)

def render_comparison_results(comparison_results):
    """Render model comparison results"""
    
    st.markdown("#### Comparison Results")
    
    # Performance metrics comparison
    metrics_df = pd.DataFrame(comparison_results['metrics'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Metrics comparison chart
        fig = go.Figure()
        
        metrics = ['accuracy', 'f1_score', 'precision', 'recall']
        model1_values = [metrics_df.loc[metrics_df['model'] == comparison_results['model1'], metric].values[0] for metric in metrics]
        model2_values = [metrics_df.loc[metrics_df['model'] == comparison_results['model2'], metric].values[0] for metric in metrics]
        
        fig.add_trace(go.Scatterpolar(
            r=model1_values,
            theta=metrics,
            fill='toself',
            name=comparison_results['model1']
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=model2_values,
            theta=metrics,
            fill='toself',
            name=comparison_results['model2']
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Metrics table
        st.dataframe(metrics_df, use_container_width=True)

def render_analysis_visualization(analysis_data, analysis_type, visualization_type):
    """Render analysis visualization"""
    
    st.markdown(f"#### {analysis_type}")
    
    if visualization_type == "Histogram":
        fig = px.histogram(
            analysis_data, 
            x='values', 
            title=f"{analysis_type} Distribution",
            color_discrete_sequence=['#00ff88']
        )
    elif visualization_type == "Box Plot":
        fig = px.box(
            analysis_data, 
            y='values', 
            title=f"{analysis_type} Box Plot",
            color_discrete_sequence=['#00ff88']
        )
    else:
        # Default scatter plot
        fig = px.scatter(
            analysis_data, 
            x='x', 
            y='y', 
            title=analysis_type,
            color_discrete_sequence=['#00ff88']
        )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)

def simulate_inference(image):
    """Simulate inference results for demonstration"""
    classes = ["Healthy", "Blight", "Rust", "Spot", "Mosaic"]
    confidences = np.random.dirichlet(np.ones(5) * 2)  # More realistic distribution
    
    predictions = [
        {"class": classes[i], "confidence": float(confidences[i])}
        for i in range(len(classes))
    ]
    predictions.sort(key=lambda x: x['confidence'], reverse=True)
    
    return {"predictions": predictions}

def simulate_batch_processing():
    """Simulate batch processing results"""
    return {
        "total_images": 150,
        "processed": 150,
        "avg_confidence": 0.847,
        "processing_time": 23.5,
        "class_distribution": {
            "Healthy": 45,
            "Blight": 32,
            "Rust": 28,
            "Spot": 25,
            "Mosaic": 20
        }
    }

def simulate_model_comparison(model1, model2):
    """Simulate model comparison results"""
    metrics_data = []
    
    # Model 1 metrics (slightly better)
    metrics_data.append({
        "model": model1,
        "accuracy": 0.91,
        "f1_score": 0.89,
        "precision": 0.92,
        "recall": 0.88
    })
    
    # Model 2 metrics
    metrics_data.append({
        "model": model2,
        "accuracy": 0.87,
        "f1_score": 0.85,
        "precision": 0.86,
        "recall": 0.84
    })
    
    return {
        "model1": model1,
        "model2": model2,
        "metrics": metrics_data
    }

def generate_analysis_data(analysis_type):
    """Generate mock analysis data"""
    if analysis_type == "Prediction Confidence Distribution":
        return pd.DataFrame({
            'values': np.random.beta(2, 1, 100)  # Skewed towards higher confidence
        })
    else:
        # Generic data for other analysis types
        return pd.DataFrame({
            'x': np.random.randn(100),
            'y': np.random.randn(100),
            'values': np.random.randn(100)
        })