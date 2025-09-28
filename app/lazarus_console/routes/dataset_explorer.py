"""
Dataset Explorer Route - Lazarus Console
Interactive dataset exploration with clickable charts and class analysis
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from PIL import Image
from components.state_manager import get_state, set_state
from utils.theme import create_alert

def render_dataset_explorer():
    """Render dataset exploration interface"""
    
    st.markdown("## 📊 Dataset Explorer")
    st.markdown("*Interactive analysis of plant disease dataset*")
    
    dataset_manager = st.session_state.get('dataset_manager')
    
    if not dataset_manager:
        st.error("Dataset manager not initialized")
        return
    
    manifest = dataset_manager.get_manifest()
    if manifest is None:
        st.warning("No dataset loaded. Scanning data directory...")
        dataset_manager.refresh_manifest()
        st.rerun()
        return
    
    # Dataset overview
    render_dataset_stats(dataset_manager)
    
    # Main explorer interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        render_class_distribution(dataset_manager)
        render_resolution_analysis(dataset_manager)
    
    with col2:
        render_sample_gallery(dataset_manager)
        render_balance_analysis(dataset_manager)

def render_dataset_stats(dataset_manager):
    """Render high-level dataset statistics"""
    
    class_stats = dataset_manager.get_class_statistics()
    image_stats = dataset_manager.get_image_statistics()
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Images", f"{class_stats.get('total_images', 0):,}")
    
    with col2:
        st.metric("Classes", class_stats.get('num_classes', 0))
    
    with col3:
        st.metric("Avg Resolution", f"{image_stats.get('width_stats', {}).get('mean', 0):.0f}px")
    
    with col4:
        st.metric("Total Size", f"{image_stats.get('file_size_stats', {}).get('total_gb', 0):.1f} GB")
    
    with col5:
        balance_ratio = class_stats.get('imbalance_ratio', 1.0)
        balance_status = "⚖️" if balance_ratio < 3 else "⚠️" if balance_ratio < 10 else "❌"
        st.metric("Balance Ratio", f"{balance_ratio:.1f}x", delta=balance_status)

def render_class_distribution(dataset_manager):
    """Render interactive class distribution chart"""
    
    st.markdown("### Class Distribution")
    
    class_stats = dataset_manager.get_class_statistics()
    class_counts = class_stats.get('class_counts', {})
    
    if not class_counts:
        st.warning("No class data available")
        return
    
    # Prepare data for visualization
    chart_data = []
    for class_name, count in class_counts.items():
        display_name = class_name.split('___')[-1] if '___' in class_name else class_name
        chart_data.append({
            'Class': display_name,
            'Full_Name': class_name,
            'Count': count,
            'Percentage': count / sum(class_counts.values()) * 100
        })
    
    df = pd.DataFrame(chart_data).sort_values('Count', ascending=False)
    
    # Interactive bar chart
    fig = px.bar(
        df,
        x='Class',
        y='Count',
        hover_data=['Full_Name', 'Percentage'],
        title="Sample Count by Class",
        color='Count',
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        height=400,
        xaxis_tickangle=-45,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        title_font_size=16
    )
    
    # Handle chart clicks
    selected_chart = st.plotly_chart(fig, use_container_width=True, key="class_distribution_chart")
    
    # Class selector for detailed view
    selected_class = st.selectbox(
        "Focus on specific class:",
        ['All Classes'] + list(class_counts.keys()),
        key="class_selector"
    )
    
    if selected_class != 'All Classes':
        set_state('selected_class', selected_class)
        render_class_details(dataset_manager, selected_class)

def render_class_details(dataset_manager, class_name):
    """Render detailed analysis for specific class"""
    
    st.markdown(f"#### Class Analysis: {class_name}")
    
    manifest = dataset_manager.get_manifest()
    class_data = manifest[manifest['class_name'] == class_name]
    
    if len(class_data) == 0:
        st.warning("No data found for this class")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Sample Count", len(class_data))
        st.metric("Avg Width", f"{class_data['width'].mean():.0f}px")
    
    with col2:
        st.metric("Avg Height", f"{class_data['height'].mean():.0f}px")
        st.metric("Avg File Size", f"{class_data['file_size'].mean() / (1024*1024):.1f} MB")
    
    with col3:
        st.metric("Aspect Ratio", f"{class_data['aspect_ratio'].mean():.2f}")
        st.metric("Resolution Std", f"{class_data['width'].std():.0f}px")
    
    # Augmentation suggestions
    suggestions = dataset_manager.get_augmentation_suggestions(class_name)
    
    with st.expander("Augmentation Suggestions", expanded=False):
        for suggestion in suggestions[:5]:  # Show top 5
            st.info(suggestion)

def render_resolution_analysis(dataset_manager):
    """Render resolution distribution analysis"""
    
    st.markdown("### Resolution Analysis")
    
    manifest = dataset_manager.get_manifest()
    image_stats = dataset_manager.get_image_statistics()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Resolution distribution
        fig = px.scatter(
            manifest,
            x='width',
            y='height',
            color='class_name',
            title="Resolution Distribution",
            hover_data=['filename'],
            opacity=0.6
        )
        fig.update_layout(
            height=300,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Aspect ratio distribution
        fig = px.histogram(
            manifest,
            x='aspect_ratio',
            title="Aspect Ratio Distribution",
            nbins=30,
            color_discrete_sequence=['#1f77b4']
        )
        fig.update_layout(
            height=300,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Common resolutions
    common_resolutions = image_stats.get('common_resolutions', {})
    if common_resolutions:
        st.markdown("#### Most Common Resolutions")
        res_data = [
            {'Resolution': f"{w}x{h}", 'Count': count}
            for (w, h), count in list(common_resolutions.items())[:5]
        ]
        st.dataframe(pd.DataFrame(res_data), use_container_width=True)

def render_sample_gallery(dataset_manager):
    """Render sample image gallery"""
    
    st.markdown("### Sample Gallery")
    
    selected_class = get_state('selected_class', None)
    
    # Get samples
    if selected_class and selected_class != 'All Classes':
        samples = dataset_manager.get_sample_images(selected_class, num_samples=6)
    else:
        samples = dataset_manager.get_sample_images(num_samples=6)
    
    if not samples:
        st.warning("No sample images available")
        return
    
    # Display samples in grid
    cols = st.columns(2)
    
    for i, sample in enumerate(samples):
        col_idx = i % 2
        
        with cols[col_idx]:
            try:
                img_path = sample['path']
                
                # Load and display image
                img = Image.open(img_path)
                
                # Resize for display
                img.thumbnail((300, 300), Image.Resampling.LANCZOS)
                
                st.image(img, caption=f"{sample['class_name']}", use_column_width=True)
                
                # Image details
                with st.expander(f"Details: {sample['filename']}", expanded=False):
                    st.text(f"Resolution: {sample['width']}x{sample['height']}")
                    st.text(f"Size: {sample['size_mb']:.2f} MB")
                    st.text(f"Class: {sample['class_name']}")
                
            except Exception as e:
                st.error(f"Error loading image: {e}")
    
    # Refresh gallery
    if st.button("Refresh Gallery", key="refresh_gallery"):
        st.rerun()

def render_balance_analysis(dataset_manager):
    """Render class balance analysis"""
    
    st.markdown("### Balance Analysis")
    
    balance_analysis = dataset_manager.analyze_class_balance()
    
    if not balance_analysis or 'error' in balance_analysis:
        st.warning("Balance analysis unavailable")
        return
    
    # Balance status
    balance_status = balance_analysis.get('balance_status', 'unknown')
    imbalance_severity = balance_analysis.get('imbalance_severity', 'unknown')
    
    if balance_status == 'balanced':
        st.success(f"✅ Dataset is well-balanced")
    elif balance_status == 'moderately_imbalanced':
        st.warning(f"⚠️ Dataset is moderately imbalanced")
    else:
        st.error(f"❌ Dataset is severely imbalanced")
    
    # Recommendations
    recommendations = balance_analysis.get('recommendations', [])
    if recommendations:
        st.markdown("#### Recommendations")
        for rec in recommendations:
            st.info(rec)
    
    # Suggested actions
    suggested_actions = balance_analysis.get('suggested_actions', [])
    if suggested_actions:
        st.markdown("#### Quick Actions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'data_augmentation' in suggested_actions:
                if st.button("Apply Data Augmentation", key="apply_augmentation"):
                    st.info("Data augmentation configuration would be applied")
        
        with col2:
            if 'class_weighting' in suggested_actions:
                if st.button("Configure Class Weights", key="config_weights"):
                    st.info("Class weighting configuration would be applied")
    
    # Export analysis
    if st.button("Export Dataset Analysis", key="export_analysis"):
        analysis_report = dataset_manager.export_analysis_report()
        
        # Convert to JSON string for download
        import json
        report_json = json.dumps(analysis_report, indent=2, default=str)
        
        st.download_button(
            "Download Analysis Report",
            data=report_json,
            file_name=f"dataset_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )