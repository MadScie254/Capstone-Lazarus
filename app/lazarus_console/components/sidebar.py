"""
Sidebar Navigation Component (Minimal/Hidden by default)
Used for expandable context panels and drawers
"""

import streamlit as st
from components.state_manager import get_state, set_state

def render_navigation():
    """Render sidebar navigation (minimal by default)"""
    
    # Keep sidebar minimal since we use top navigation
    with st.sidebar:
        st.markdown("### Quick Actions")
        
        if st.button("Refresh Models"):
            # Trigger model refresh
            model_manager = st.session_state.get('model_manager')
            if model_manager:
                model_manager.refresh_available_models()
            st.rerun()
        
        if st.button("Clear Cache"):
            from components.state_manager import clear_cache_keys
            clear_cache_keys()
            st.success("Cache cleared")
        
        if st.button("Reset State"):
            from components.state_manager import reset_state
            reset_state()
            st.rerun()
        
        # System info
        st.markdown("### System Info")
        system_status = get_state('system_status', {})
        
        if system_status.get('gpu_available', False):
            st.metric("GPU", system_status.get('gpu_name', 'Unknown'))
            st.metric("VRAM", f"{system_status.get('vram_usage', 0):.1f}GB")
        
        st.metric("CPU", f"{system_status.get('cpu_usage', 0):.1f}%")
        st.metric("Memory", f"{system_status.get('memory_usage', 0):.1f}%")

def render_contextual_drawer(content_type: str = None):
    """Render contextual drawer content based on current context"""
    
    current_route = get_state('current_route', 'home')
    
    with st.sidebar:
        st.markdown(f"### {current_route.title()} Context")
        
        if current_route == 'dataset':
            render_dataset_context()
        elif current_route == 'training':
            render_training_context()
        elif current_route == 'inference':
            render_inference_context()
        elif current_route == 'explain':
            render_explainability_context()
        elif current_route == 'compare':
            render_compare_context()
        else:
            st.info("No contextual information available")

def render_dataset_context():
    """Dataset-specific sidebar content"""
    
    dataset_manager = st.session_state.get('dataset_manager')
    if not dataset_manager:
        st.warning("Dataset manager not initialized")
        return
    
    # Quick stats
    manifest = get_state('dataset_manifest')
    if manifest is not None:
        st.metric("Total Images", len(manifest))
        st.metric("Classes", manifest['class_name'].nunique())
        st.metric("Avg per Class", f"{len(manifest) // manifest['class_name'].nunique()}")
    
    # Quick filters
    st.markdown("#### Quick Filters")
    
    if st.button("Show Imbalanced Classes"):
        set_state('show_imbalance_alert', True)
    
    if st.button("Highlight Small Classes"):
        set_state('highlight_small_classes', True)

def render_training_context():
    """Training-specific sidebar content"""
    
    training_status = get_state('training_run_status', 'idle')
    
    st.metric("Status", training_status.title())
    
    if training_status == 'running':
        if st.button("Pause Training", type="secondary"):
            set_state('training_run_status', 'paused')
        
        if st.button("Stop Training", type="primary"):
            set_state('training_run_status', 'stopped')
    
    elif training_status in ['paused', 'stopped']:
        if st.button("Resume Training", type="primary"):
            set_state('training_run_status', 'running')
    
    # Quick settings
    st.markdown("#### Quick Settings")
    
    learning_rate = st.number_input("Learning Rate", value=0.001, format="%.4f", key="sidebar_lr")
    batch_size = st.number_input("Batch Size", value=32, min_value=1, max_value=128, key="sidebar_batch")
    
    if st.button("Apply Settings"):
        set_state('training_lr', learning_rate)
        set_state('training_batch_size', batch_size)

def render_inference_context():
    """Inference-specific sidebar content"""
    
    # Batch results summary
    batch_results = get_state('batch_results')
    if batch_results:
        st.metric("Processed", len(batch_results))
        
        # Confidence distribution
        confidences = [r.get('confidence', 0) for r in batch_results]
        if confidences:
            avg_conf = sum(confidences) / len(confidences)
            st.metric("Avg Confidence", f"{avg_conf:.3f}")
    
    # Quick actions
    st.markdown("#### Quick Actions")
    
    if st.button("Export Results"):
        set_state('export_results_requested', True)
    
    if st.button("Generate Report"):
        set_state('generate_report_requested', True)

def render_explainability_context():
    """Explainability-specific sidebar content"""
    
    # Grad-CAM settings
    st.markdown("#### Grad-CAM Settings")
    
    layer_name = st.selectbox("Target Layer", 
                             ["last_conv", "mixed7", "block6a", "features"], 
                             key="gradcam_layer")
    
    blend_alpha = st.slider("Overlay Blend", 0.0, 1.0, 0.5, key="gradcam_blend")
    
    colormap = st.selectbox("Colormap", ["jet", "hot", "viridis", "plasma"], key="gradcam_colormap")
    
    if st.button("Update Settings"):
        set_state('gradcam_layer', layer_name)
        set_state('gradcam_blend', blend_alpha)
        set_state('gradcam_colormap', colormap)

def render_compare_context():
    """Model comparison specific sidebar content"""
    
    available_models = get_state('available_models', [])
    
    if len(available_models) >= 2:
        st.markdown("#### Comparison Settings")
        
        model_a = st.selectbox("Model A", available_models, key="compare_model_a")
        model_b = st.selectbox("Model B", available_models, key="compare_model_b")
        
        comparison_metrics = st.multiselect("Metrics", 
                                          ["accuracy", "f1_score", "precision", "recall", "latency"],
                                          default=["accuracy", "f1_score", "latency"],
                                          key="compare_metrics")
        
        if st.button("Run Comparison"):
            set_state('comparison_models', [model_a, model_b])
            set_state('comparison_metrics', comparison_metrics)
            set_state('run_comparison_requested', True)
    
    else:
        st.warning("Need at least 2 models for comparison")