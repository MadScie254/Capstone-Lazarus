"""
Model Compare Route - Lazarus Console
Advanced model comparison and benchmarking tools
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime
from components.state_manager import get_state, set_state, add_decision_log_entry
from utils.model_manager import ModelManager

def render_model_compare():
    """Render model comparison interface"""
    
    st.markdown("## ⚖️ Model Compare")
    st.markdown("*Advanced model comparison and benchmarking*")
    
    # Initialize model manager
    if 'model_manager' not in st.session_state:
        st.session_state.model_manager = ModelManager()
    
    # Comparison controls
    render_comparison_controls()
    
    # Main comparison interface
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Performance", "⚡ Speed", "💾 Resources", "🎯 Deployment"])
    
    with tab1:
        render_performance_comparison()
    
    with tab2:
        render_speed_comparison()
    
    with tab3:
        render_resource_comparison()
    
    with tab4:
        render_deployment_comparison()

def render_comparison_controls():
    """Render comparison control panel"""
    
    st.markdown("### Model Selection")
    
    # Available models
    available_models = [
        "EfficientNet-B0", "EfficientNet-B1", "ResNet-50", "ResNet-101",
        "MobileNet-V2", "MobileNet-V3", "Vision Transformer", "DenseNet-121"
    ]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        model1 = st.selectbox("Model 1", available_models, index=0, key="compare_model1")
        set_state('compare_model1', model1)
    
    with col2:
        model2 = st.selectbox("Model 2", available_models, index=1, key="compare_model2")
        set_state('compare_model2', model2)
    
    with col3:
        model3 = st.selectbox("Model 3 (Optional)", ["None"] + available_models, index=0, key="compare_model3")
        if model3 != "None":
            set_state('compare_model3', model3)
    
    with col4:
        test_dataset = st.selectbox("Test Dataset", ["Validation Set", "Test Set", "Holdout Set"], index=0)
        set_state('comparison_dataset', test_dataset)
    
    # Comparison options
    st.markdown("### Comparison Options")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        include_ensemble = st.checkbox("Include Ensemble", value=False)
        set_state('include_ensemble', include_ensemble)
    
    with col2:
        benchmark_speed = st.checkbox("Benchmark Speed", value=True)
        set_state('benchmark_speed', benchmark_speed)
    
    with col3:
        memory_profiling = st.checkbox("Memory Profiling", value=True)
        set_state('memory_profiling', memory_profiling)
    
    with col4:
        export_results = st.checkbox("Export Results", value=False)
        set_state('export_results', export_results)
    
    # Run comparison
    if st.button("🔄 Run Comparison", type="primary", use_container_width=True):
        with st.spinner("Running comprehensive comparison..."):
            comparison_results = run_model_comparison()
            set_state('comparison_results', comparison_results)
            add_decision_log_entry("model_comparison", {
                "models": [model1, model2, model3] if model3 != "None" else [model1, model2],
                "timestamp": datetime.now().isoformat()
            })
            st.success("Comparison completed!")
            st.rerun()

def render_performance_comparison():
    """Render performance metrics comparison"""
    
    st.markdown("### Performance Metrics")
    
    comparison_results = get_state('comparison_results')
    if not comparison_results:
        st.info("Run a comparison to see performance metrics")
        return
    
    # Performance metrics table
    metrics_df = pd.DataFrame(comparison_results['performance_metrics'])
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Metrics Table")
        
        # Style the dataframe
        styled_df = metrics_df.style.format({
            'Accuracy': '{:.3f}',
            'F1-Score': '{:.3f}',
            'Precision': '{:.3f}',
            'Recall': '{:.3f}',
            'AUC-ROC': '{:.3f}'
        }).highlight_max(axis=0, color='lightgreen').highlight_min(axis=0, color='lightcoral')
        
        st.dataframe(styled_df, use_container_width=True)
    
    with col2:
        st.markdown("#### Performance Radar")
        render_performance_radar(metrics_df)
    
    # Class-wise performance
    st.markdown("#### Class-wise Performance")
    render_classwise_performance(comparison_results['classwise_metrics'])
    
    # Confusion matrices
    st.markdown("#### Confusion Matrices")
    render_confusion_matrices(comparison_results['confusion_matrices'])

def render_speed_comparison():
    """Render speed benchmarking results"""
    
    st.markdown("### Speed Benchmarking")
    
    comparison_results = get_state('comparison_results')
    if not comparison_results:
        st.info("Run a comparison to see speed benchmarks")
        return
    
    speed_data = comparison_results['speed_benchmarks']
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Inference time comparison
        st.markdown("#### Inference Time")
        
        models = [item['model'] for item in speed_data]
        inference_times = [item['inference_time_ms'] for item in speed_data]
        
        fig = px.bar(
            x=models,
            y=inference_times,
            title="Average Inference Time (ms)",
            color=inference_times,
            color_continuous_scale='viridis_r'
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Throughput comparison
        st.markdown("#### Throughput")
        
        throughput = [item['throughput_fps'] for item in speed_data]
        
        fig = px.bar(
            x=models,
            y=throughput,
            title="Throughput (FPS)",
            color=throughput,
            color_continuous_scale='viridis'
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Speed vs accuracy trade-off
    st.markdown("#### Speed vs Accuracy Trade-off")
    render_speed_accuracy_tradeoff(comparison_results)
    
    # Batch size impact
    st.markdown("#### Batch Size Impact")
    render_batch_size_impact()

def render_resource_comparison():
    """Render resource usage comparison"""
    
    st.markdown("### Resource Usage")
    
    comparison_results = get_state('comparison_results')
    if not comparison_results:
        st.info("Run a comparison to see resource usage")
        return
    
    resource_data = comparison_results['resource_usage']
    
    # Memory usage
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Memory Usage")
        
        models = [item['model'] for item in resource_data]
        memory_usage = [item['memory_mb'] for item in resource_data]
        
        fig = px.bar(
            x=models,
            y=memory_usage,
            title="Peak Memory Usage (MB)",
            color=memory_usage,
            color_continuous_scale='reds'
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Model Size")
        
        model_sizes = [item['model_size_mb'] for item in resource_data]
        
        fig = px.bar(
            x=models,
            y=model_sizes,
            title="Model Size (MB)",
            color=model_sizes,
            color_continuous_scale='blues'
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Resource efficiency matrix
    st.markdown("#### Resource Efficiency Matrix")
    render_efficiency_matrix(resource_data)
    
    # GPU utilization
    st.markdown("#### GPU Utilization")
    render_gpu_utilization()

def render_deployment_comparison():
    """Render deployment readiness comparison"""
    
    st.markdown("### Deployment Analysis")
    
    comparison_results = get_state('comparison_results')
    if not comparison_results:
        st.info("Run a comparison to see deployment analysis")
        return
    
    deployment_data = comparison_results['deployment_metrics']
    
    # Deployment readiness scores
    st.markdown("#### Deployment Readiness")
    render_deployment_readiness(deployment_data)
    
    # Platform compatibility
    st.markdown("#### Platform Compatibility")
    render_platform_compatibility(deployment_data)
    
    # Optimization recommendations
    st.markdown("#### Optimization Recommendations")
    render_optimization_recommendations(deployment_data)

def render_performance_radar(metrics_df):
    """Render radar chart for performance metrics"""
    
    fig = go.Figure()
    
    metrics = ['Accuracy', 'F1-Score', 'Precision', 'Recall', 'AUC-ROC']
    
    for _, row in metrics_df.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=[row[metric] for metric in metrics],
            theta=metrics,
            fill='toself',
            name=row['Model']
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_classwise_performance(classwise_data):
    """Render class-wise performance comparison"""
    
    # Convert to DataFrame for easier plotting
    df_data = []
    for model_data in classwise_data:
        for class_name, metrics in model_data['classes'].items():
            df_data.append({
                'Model': model_data['model'],
                'Class': class_name,
                'F1-Score': metrics['f1'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall']
            })
    
    df = pd.DataFrame(df_data)
    
    # Create grouped bar chart
    fig = px.bar(
        df,
        x='Class',
        y='F1-Score',
        color='Model',
        barmode='group',
        title="Class-wise F1-Score Comparison"
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)

def render_confusion_matrices(confusion_data):
    """Render confusion matrices for each model"""
    
    num_models = len(confusion_data)
    cols = st.columns(num_models)
    
    for i, (col, model_data) in enumerate(zip(cols, confusion_data)):
        with col:
            st.markdown(f"**{model_data['model']}**")
            
            # Create confusion matrix heatmap
            fig = px.imshow(
                model_data['matrix'],
                labels=dict(x="Predicted", y="Actual"),
                x=model_data['labels'],
                y=model_data['labels'],
                color_continuous_scale='blues',
                text_auto=True
            )
            fig.update_layout(
                height=300,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)

def render_speed_accuracy_tradeoff(comparison_results):
    """Render speed vs accuracy trade-off plot"""
    
    # Extract data
    models = []
    accuracies = []
    speeds = []
    
    for perf in comparison_results['performance_metrics']:
        models.append(perf['Model'])
        accuracies.append(perf['Accuracy'])
    
    for speed in comparison_results['speed_benchmarks']:
        speeds.append(1000 / speed['inference_time_ms'])  # Convert to FPS
    
    # Create scatter plot
    fig = px.scatter(
        x=speeds,
        y=accuracies,
        text=models,
        title="Speed vs Accuracy Trade-off",
        labels={'x': 'Speed (FPS)', 'y': 'Accuracy'},
        size=[100] * len(models)  # Fixed size
    )
    
    fig.update_traces(textposition="top center")
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)

def render_batch_size_impact():
    """Render batch size impact on performance"""
    
    batch_sizes = [1, 2, 4, 8, 16, 32]
    
    # Mock data for different models
    model1_times = [45, 25, 15, 12, 10, 9]
    model2_times = [52, 30, 18, 14, 12, 11]
    model3_times = [38, 22, 13, 10, 8, 7]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=batch_sizes, y=model1_times, mode='lines+markers', name='Model 1'))
    fig.add_trace(go.Scatter(x=batch_sizes, y=model2_times, mode='lines+markers', name='Model 2'))
    fig.add_trace(go.Scatter(x=batch_sizes, y=model3_times, mode='lines+markers', name='Model 3'))
    
    fig.update_layout(
        title="Batch Size vs Inference Time",
        xaxis_title="Batch Size",
        yaxis_title="Inference Time (ms)",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_efficiency_matrix(resource_data):
    """Render resource efficiency matrix"""
    
    models = [item['model'] for item in resource_data]
    memory = [item['memory_mb'] for item in resource_data]
    size = [item['model_size_mb'] for item in resource_data]
    
    # Normalize values for comparison
    memory_norm = [(m - min(memory)) / (max(memory) - min(memory)) for m in memory]
    size_norm = [(s - min(size)) / (max(size) - min(size)) for s in size]
    
    efficiency_data = []
    for i, model in enumerate(models):
        efficiency_data.append({
            'Model': model,
            'Memory Efficiency': 1 - memory_norm[i],  # Inverted (lower is better)
            'Size Efficiency': 1 - size_norm[i]       # Inverted (lower is better)
        })
    
    df = pd.DataFrame(efficiency_data)
    
    fig = px.bar(
        df,
        x='Model',
        y=['Memory Efficiency', 'Size Efficiency'],
        barmode='group',
        title="Resource Efficiency Comparison"
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)

def render_gpu_utilization():
    """Render GPU utilization during inference"""
    
    time_points = list(range(0, 60, 5))  # 60 seconds, every 5 seconds
    
    # Mock GPU utilization data
    model1_gpu = [20, 85, 90, 87, 88, 85, 82, 78, 75, 72, 70, 25]
    model2_gpu = [15, 92, 95, 94, 93, 91, 88, 85, 82, 80, 77, 30]
    model3_gpu = [25, 78, 82, 80, 79, 77, 75, 72, 70, 68, 65, 20]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=time_points, y=model1_gpu, mode='lines', name='Model 1', fill='tonexty'))
    fig.add_trace(go.Scatter(x=time_points, y=model2_gpu, mode='lines', name='Model 2', fill='tonexty'))
    fig.add_trace(go.Scatter(x=time_points, y=model3_gpu, mode='lines', name='Model 3', fill='tonexty'))
    
    fig.update_layout(
        title="GPU Utilization During Inference",
        xaxis_title="Time (seconds)",
        yaxis_title="GPU Utilization (%)",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_deployment_readiness(deployment_data):
    """Render deployment readiness scores"""
    
    models = [item['model'] for item in deployment_data]
    readiness_scores = [item['readiness_score'] for item in deployment_data]
    
    # Create gauge charts
    cols = st.columns(len(models))
    
    for i, (col, model, score) in enumerate(zip(cols, models, readiness_scores)):
        with col:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': model},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig.update_layout(height=250, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

def render_platform_compatibility(deployment_data):
    """Render platform compatibility matrix"""
    
    platforms = ['CPU', 'GPU', 'Mobile', 'Edge', 'Cloud']
    compatibility_matrix = []
    
    for item in deployment_data:
        model_compat = []
        for platform in platforms:
            # Mock compatibility scores
            score = np.random.uniform(0.3, 1.0)
            model_compat.append(score)
        compatibility_matrix.append(model_compat)
    
    models = [item['model'] for item in deployment_data]
    
    fig = px.imshow(
        compatibility_matrix,
        labels=dict(x="Platform", y="Model"),
        x=platforms,
        y=models,
        color_continuous_scale='RdYlGn',
        title="Platform Compatibility Matrix"
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)

def render_optimization_recommendations(deployment_data):
    """Render optimization recommendations"""
    
    for item in deployment_data:
        with st.expander(f"Recommendations for {item['model']}"):
            recommendations = item.get('recommendations', [])
            
            if not recommendations:
                recommendations = [
                    "Consider quantization for faster inference",
                    "Batch processing can improve throughput",
                    "Model pruning may reduce memory usage",
                    "ONNX conversion recommended for deployment"
                ]
            
            for i, rec in enumerate(recommendations, 1):
                st.write(f"{i}. {rec}")

def run_model_comparison():
    """Run comprehensive model comparison"""
    
    model1 = get_state('compare_model1')
    model2 = get_state('compare_model2')
    model3 = get_state('compare_model3')
    
    models = [model1, model2]
    if model3:
        models.append(model3)
    
    # Generate mock comparison results
    results = {
        'performance_metrics': generate_mock_performance_metrics(models),
        'speed_benchmarks': generate_mock_speed_benchmarks(models),
        'resource_usage': generate_mock_resource_usage(models),
        'deployment_metrics': generate_mock_deployment_metrics(models),
        'classwise_metrics': generate_mock_classwise_metrics(models),
        'confusion_matrices': generate_mock_confusion_matrices(models)
    }
    
    return results

def generate_mock_performance_metrics(models):
    """Generate mock performance metrics"""
    metrics = []
    base_scores = [0.91, 0.87, 0.93]  # Different base performance levels
    
    for i, model in enumerate(models):
        base = base_scores[i % len(base_scores)]
        metrics.append({
            'Model': model,
            'Accuracy': base + np.random.normal(0, 0.01),
            'F1-Score': base - 0.02 + np.random.normal(0, 0.01),
            'Precision': base + 0.01 + np.random.normal(0, 0.01),
            'Recall': base - 0.01 + np.random.normal(0, 0.01),
            'AUC-ROC': base + 0.03 + np.random.normal(0, 0.005)
        })
    
    return metrics

def generate_mock_speed_benchmarks(models):
    """Generate mock speed benchmarks"""
    benchmarks = []
    base_times = [25, 35, 18]  # Different speed characteristics
    
    for i, model in enumerate(models):
        base_time = base_times[i % len(base_times)]
        benchmarks.append({
            'model': model,
            'inference_time_ms': base_time + np.random.normal(0, 2),
            'throughput_fps': 1000 / (base_time + np.random.normal(0, 2)),
            'batch_size': 8,
            'platform': 'GPU'
        })
    
    return benchmarks

def generate_mock_resource_usage(models):
    """Generate mock resource usage data"""
    usage = []
    base_memory = [1200, 2400, 800]  # Different memory requirements
    base_size = [94, 178, 52]        # Different model sizes
    
    for i, model in enumerate(models):
        usage.append({
            'model': model,
            'memory_mb': base_memory[i % len(base_memory)] + np.random.normal(0, 50),
            'model_size_mb': base_size[i % len(base_size)] + np.random.normal(0, 5),
            'gpu_memory_mb': base_memory[i % len(base_memory)] * 0.8 + np.random.normal(0, 30),
            'parameters_m': base_size[i % len(base_size)] / 4  # Rough estimate
        })
    
    return usage

def generate_mock_deployment_metrics(models):
    """Generate mock deployment metrics"""
    metrics = []
    
    for model in models:
        metrics.append({
            'model': model,
            'readiness_score': np.random.uniform(0.7, 0.95),
            'optimization_score': np.random.uniform(0.6, 0.9),
            'compatibility_score': np.random.uniform(0.8, 1.0),
            'recommendations': [
                "Consider quantization for faster inference",
                "Batch processing recommended for throughput",
                "Model serves well on edge devices"
            ]
        })
    
    return metrics

def generate_mock_classwise_metrics(models):
    """Generate mock class-wise metrics"""
    classes = ['Healthy', 'Blight', 'Rust', 'Spot', 'Mosaic']
    classwise = []
    
    for model in models:
        model_classes = {}
        for cls in classes:
            model_classes[cls] = {
                'precision': np.random.uniform(0.8, 0.95),
                'recall': np.random.uniform(0.8, 0.95),
                'f1': np.random.uniform(0.8, 0.95)
            }
        
        classwise.append({
            'model': model,
            'classes': model_classes
        })
    
    return classwise

def generate_mock_confusion_matrices(models):
    """Generate mock confusion matrices"""
    classes = ['Healthy', 'Blight', 'Rust', 'Spot', 'Mosaic']
    matrices = []
    
    for model in models:
        # Generate a reasonably realistic confusion matrix
        matrix = np.random.randint(5, 25, (5, 5))
        
        # Make diagonal elements larger (correct predictions)
        for i in range(5):
            matrix[i, i] = np.random.randint(80, 95)
        
        matrices.append({
            'model': model,
            'matrix': matrix.tolist(),
            'labels': classes
        })
    
    return matrices