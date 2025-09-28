"""
Lazarus Console - AI Plant Disease Diagnostics Mission Control
A cinematic, production-grade Streamlit dashboard for dataset exploration, 
training monitoring, inference, explainability, and deployment assessment.
"""

import streamlit as st
import sys
from pathlib import Path
import json
import time
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import dashboard components
from components.header import render_mission_header
from components.sidebar import render_navigation
from components.state_manager import initialize_global_state, get_state, set_state
from utils.theme import apply_dark_theme, inject_custom_css
from utils.model_manager import ModelManager
from utils.dataset_manager import DatasetManager

# Import route modules
from routes.home import render_home
from routes.dataset_explorer import render_dataset_explorer
from routes.training_monitor import render_training_monitor
from routes.inference_lab import render_inference_lab
from routes.explainability_studio import render_explainability_studio
from routes.model_compare import render_model_compare
from routes.system_profiler import render_system_profiler
from routes.settings import render_settings

def main():
    """Main application entry point with enhanced loading and error handling"""
    
    # Configure page
    st.set_page_config(
        page_title="Lazarus Console - AI Plant Disease Diagnostics",
        page_icon="🌱",
        layout="wide",
        initial_sidebar_state="collapsed",
        menu_items={
            'Get Help': None,
            'Report a Bug': None,
            'About': "Lazarus Console - AI Plant Disease Diagnostics Mission Control"
        }
    )
    
    # Apply theme and styling
    apply_dark_theme()
    inject_custom_css()
    
    # Initialize global state
    initialize_global_state()
    
    # Initialize managers with loading indicators
    with st.spinner("🚀 Initializing Lazarus Console..."):
        if 'model_manager' not in st.session_state:
            try:
                st.session_state.model_manager = ModelManager(project_root)
                st.success("✅ Model Manager initialized", icon="🤖")
            except Exception as e:
                st.error(f"❌ Model Manager failed: {str(e)}")
                st.session_state.model_manager = None
        
        if 'dataset_manager' not in st.session_state:
            try:
                st.session_state.dataset_manager = DatasetManager(project_root)
                st.success("✅ Dataset Manager initialized", icon="📊")
            except Exception as e:
                st.error(f"❌ Dataset Manager failed: {str(e)}")
                st.session_state.dataset_manager = None
    
    # Render mission header
    render_mission_header()
    
    # Get current route
    current_route = get_state('current_route', 'home')
    
    # Route handling with better error management
    route_map = {
        'home': render_home,
        'dataset': render_dataset_explorer,
        'training': render_training_monitor,
        'inference': render_inference_lab,
        'explain': render_explainability_studio,
        'compare': render_model_compare,
        'profiler': render_system_profiler,
        'settings': render_settings
    }
    
    # Render current route with enhanced error handling
    if current_route in route_map:
        try:
            with st.container():
                route_map[current_route]()
        except Exception as e:
            st.error(f"🚨 Error in {current_route} module: {str(e)}")
            with st.expander("🔍 Error Details", expanded=False):
                st.exception(e)
            
            # Fallback to home
            st.info("🏠 Redirecting to Mission Control...")
            set_state('current_route', 'home')
            time.sleep(2)
            st.rerun()
    else:
        st.error(f"❌ Unknown route: {current_route}")
        set_state('current_route', 'home')
        st.rerun()

if __name__ == "__main__":
    main()