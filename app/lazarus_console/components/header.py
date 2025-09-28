"""
Mission Header Component for Lazarus Console
Persistent header with navigation, status, and system controls
"""

import streamlit as st
import time
from typing import Dict, Any
from components.state_manager import get_state, set_state, update_system_status
from utils.theme import create_status_metric

def render_mission_header():
    """Render the mission control header with navigation and status"""
    
    # Update system status
    update_system_status()
    
    # Get current state
    current_route = get_state('current_route', 'home')
    selected_model = get_state('selected_model', 'No Model')
    dataset_version = get_state('dataset_version', 'v1.0')
    system_status = get_state('system_status', {})
    
    # Get model performance metrics
    model_manager = st.session_state.get('model_manager')
    model_metrics = {}
    if model_manager and selected_model and selected_model != 'No Model':
        model_metrics = model_manager.get_model_metrics(selected_model)
    
    # Header container
    header_html = f"""
    <div class="mission-header">
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
            <div>
                <h1 class="mission-title">LAZARUS CONSOLE</h1>
                <div style="color: var(--muted-text); font-size: 0.875rem; margin-top: 0.5rem;">
                    AI Plant Disease Diagnostics Mission Control
                </div>
            </div>
            <div style="text-align: right; color: var(--muted-text); font-size: 0.75rem;">
                Session: {get_state('session_id', 'Unknown')}<br>
                Model: {selected_model}<br>
                Dataset: {dataset_version}
            </div>
        </div>
        
        <div class="status-strip">
            {create_status_metric("F1 Score", f"{model_metrics.get('macro_f1', 0.0):.3f}", get_metric_status(model_metrics.get('macro_f1', 0.0), 0.85))}
            {create_status_metric("Critical Recall", f"{model_metrics.get('critical_recall', 0.0):.3f}", get_metric_status(model_metrics.get('critical_recall', 0.0), 0.90))}
            {create_status_metric("Latency", f"{model_metrics.get('latency_ms', 0):.0f}ms", get_latency_status(model_metrics.get('latency_ms', 0)))}
            {create_status_metric("Model Size", f"{model_metrics.get('size_mb', 0):.1f}MB", get_size_status(model_metrics.get('size_mb', 0)))}
            {create_status_metric("GPU", f"{system_status.get('vram_usage', 0):.1f}GB", get_vram_status(system_status.get('vram_percent', 0)))}
        </div>
        
        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 1rem;">
            <div class="nav-container" style="flex: 1; margin-right: 2rem;">
                {render_navigation_buttons(current_route)}
            </div>
            <div style="display: flex; gap: 1rem; align-items: center;">
                {render_toggle_switches()}
            </div>
        </div>
    </div>
    """
    
    st.markdown(header_html, unsafe_allow_html=True)

def render_navigation_buttons(current_route: str) -> str:
    """Render navigation buttons"""
    
    routes = [
        ('home', 'Mission Control'),
        ('dataset', 'Dataset'),
        ('training', 'Training'),
        ('inference', 'Inference'),
        ('explain', 'Explain'),
        ('compare', 'Compare'),
        ('profiler', 'Profiler'),
        ('settings', 'Settings')
    ]
    
    nav_html = ""
    for route_key, route_name in routes:
        active_class = "active" if route_key == current_route else ""
        nav_html += f"""
        <div class="nav-button {active_class}" onclick="setRoute('{route_key}')">
            {route_name}
        </div>
        """
    
    # Add JavaScript for navigation
    nav_html += """
    <script>
    function setRoute(route) {
        // This will be handled by Streamlit's rerun mechanism
        window.parent.postMessage({type: 'SET_ROUTE', route: route}, '*');
    }
    </script>
    """
    
    return nav_html

def render_toggle_switches() -> str:
    """Render AMP/ONNX toggle switches"""
    
    amp_enabled = get_state('amp_enabled', False)
    onnx_enabled = get_state('onnx_enabled', False)
    
    amp_active = "active" if amp_enabled else ""
    onnx_active = "active" if onnx_enabled else ""
    
    return f"""
    <div style="display: flex; align-items: center; gap: 1.5rem;">
        <div class="toggle-container">
            <span style="color: var(--muted-text); font-size: 0.875rem;">AMP</span>
            <div class="toggle-switch {amp_active}" onclick="toggleAMP()">
                <div class="toggle-knob"></div>
            </div>
        </div>
        <div class="toggle-container">
            <span style="color: var(--muted-text); font-size: 0.875rem;">ONNX</span>
            <div class="toggle-switch {onnx_active}" onclick="toggleONNX()">
                <div class="toggle-knob"></div>
            </div>
        </div>
    </div>
    
    <script>
    function toggleAMP() {{
        window.parent.postMessage({{type: 'TOGGLE_AMP'}}, '*');
    }}
    
    function toggleONNX() {{
        window.parent.postMessage({{type: 'TOGGLE_ONNX'}}, '*');
    }}
    </script>
    """

def handle_header_interactions():
    """Handle header interaction events"""
    
    # Check for route changes via query params or custom events
    query_params = st.experimental_get_query_params()
    
    if 'route' in query_params:
        new_route = query_params['route'][0]
        current_route = get_state('current_route')
        
        if new_route != current_route and new_route in ['home', 'dataset', 'training', 'inference', 'explain', 'compare', 'profiler', 'settings']:
            set_state('previous_route', current_route)
            set_state('current_route', new_route)
            st.experimental_set_query_params()  # Clear query params
            st.rerun()

def create_navigation_columns():
    """Create column-based navigation as fallback"""
    
    current_route = get_state('current_route', 'home')
    
    col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)
    
    routes = [
        ('home', 'Mission', col1),
        ('dataset', 'Dataset', col2),
        ('training', 'Training', col3),
        ('inference', 'Inference', col4),
        ('explain', 'Explain', col5),
        ('compare', 'Compare', col6),
        ('profiler', 'Profiler', col7),
        ('settings', 'Settings', col8)
    ]
    
    for route_key, route_name, col in routes:
        with col:
            button_type = "primary" if route_key == current_route else "secondary"
            if st.button(route_name, key=f"nav_{route_key}", type=button_type):
                set_state('previous_route', current_route)
                set_state('current_route', route_key)
                st.rerun()

def create_toggle_columns():
    """Create column-based toggles as fallback"""
    
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    
    with col1:
        amp_enabled = get_state('amp_enabled', False)
        if st.checkbox("AMP", value=amp_enabled, key="amp_toggle"):
            set_state('amp_enabled', True)
            # Clear relevant caches
            from components.state_manager import clear_cache_keys
            clear_cache_keys("pytorch")
        else:
            set_state('amp_enabled', False)
    
    with col2:
        onnx_enabled = get_state('onnx_enabled', False)
        if st.checkbox("ONNX", value=onnx_enabled, key="onnx_toggle"):
            set_state('onnx_enabled', True)
            # Clear relevant caches
            from components.state_manager import clear_cache_keys
            clear_cache_keys("pytorch")
        else:
            set_state('onnx_enabled', False)
    
    with col3:
        precision = get_state('precision', 'fp32')
        new_precision = st.selectbox("Precision", ['fp32', 'fp16'], 
                                   index=0 if precision == 'fp32' else 1,
                                   key="precision_select")
        if new_precision != precision:
            set_state('precision', new_precision)
    
    with col4:
        batch_size = get_state('batch_size', 8)
        new_batch_size = st.number_input("Batch Size", min_value=1, max_value=32, 
                                       value=batch_size, key="batch_size_input")
        if new_batch_size != batch_size:
            set_state('batch_size', new_batch_size)

def get_metric_status(value: float, threshold: float) -> str:
    """Get status color for metrics"""
    if value >= threshold:
        return "success"
    elif value >= threshold * 0.8:
        return "warning"
    else:
        return "error"

def get_latency_status(latency_ms: float) -> str:
    """Get status color for latency"""
    if latency_ms <= 200:
        return "success"
    elif latency_ms <= 500:
        return "warning"
    else:
        return "error"

def get_size_status(size_mb: float) -> str:
    """Get status color for model size"""
    if size_mb <= 50:
        return "success"
    elif size_mb <= 100:
        return "warning"
    else:
        return "error"

def get_vram_status(vram_percent: float) -> str:
    """Get status color for VRAM usage"""
    if vram_percent <= 60:
        return "success"
    elif vram_percent <= 80:
        return "warning"
    else:
        return "error"