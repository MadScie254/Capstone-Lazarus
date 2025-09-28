"""
Settings Route - Lazarus Console
Configuration management and system preferences
"""

import streamlit as st
import json
import os
from datetime import datetime
from components.state_manager import get_state, set_state, add_decision_log_entry
from utils.theme import apply_theme

def render_settings():
    """Render settings interface"""
    
    st.markdown("## ⚙️ Settings")
    st.markdown("*Configuration and preferences management*")
    
    # Settings tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎨 Appearance", "🤖 Models", "📊 Data", "🔧 System", "📋 Advanced"])
    
    with tab1:
        render_appearance_settings()
    
    with tab2:
        render_model_settings()
    
    with tab3:
        render_data_settings()
    
    with tab4:
        render_system_settings()
    
    with tab5:
        render_advanced_settings()

def render_appearance_settings():
    """Render appearance and theme settings"""
    
    st.markdown("### Appearance Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Theme Configuration")
        
        # Theme selection
        theme_options = ["Dark Mission Control", "Light Professional", "High Contrast", "Custom"]
        current_theme = get_state('selected_theme', 'Dark Mission Control')
        selected_theme = st.selectbox("Theme", theme_options, index=theme_options.index(current_theme))
        
        if selected_theme != current_theme:
            set_state('selected_theme', selected_theme)
            st.success("Theme updated! Refresh to apply changes.")
        
        # Color scheme
        st.markdown("#### Color Scheme")
        
        primary_color = st.color_picker("Primary Color", "#00ff88")
        secondary_color = st.color_picker("Secondary Color", "#ff6b35")
        accent_color = st.color_picker("Accent Color", "#4ecdc4")
        
        set_state('primary_color', primary_color)
        set_state('secondary_color', secondary_color)
        set_state('accent_color', accent_color)
        
        # Layout options
        st.markdown("#### Layout Options")
        
        sidebar_width = st.selectbox("Sidebar Width", ["Narrow", "Normal", "Wide"], index=1)
        header_style = st.selectbox("Header Style", ["Compact", "Standard", "Extended"], index=1)
        card_style = st.selectbox("Card Style", ["Rounded", "Sharp", "Minimal"], index=0)
        
        set_state('sidebar_width', sidebar_width)
        set_state('header_style', header_style)
        set_state('card_style', card_style)
    
    with col2:
        st.markdown("#### Display Options")
        
        # Animation settings
        enable_animations = st.checkbox("Enable Animations", value=get_state('enable_animations', True))
        animation_speed = st.selectbox("Animation Speed", ["Slow", "Normal", "Fast"], index=1)
        
        set_state('enable_animations', enable_animations)
        set_state('animation_speed', animation_speed)
        
        # Chart settings
        st.markdown("#### Chart Settings")
        
        default_chart_theme = st.selectbox("Chart Theme", ["plotly_dark", "plotly", "ggplot2", "seaborn"], index=0)
        show_grid = st.checkbox("Show Grid Lines", value=get_state('show_grid', True))
        interactive_charts = st.checkbox("Interactive Charts", value=get_state('interactive_charts', True))
        
        set_state('default_chart_theme', default_chart_theme)
        set_state('show_grid', show_grid)
        set_state('interactive_charts', interactive_charts)
        
        # Font settings
        st.markdown("#### Typography")
        
        font_family = st.selectbox("Font Family", ["Inter", "Roboto", "Helvetica", "Arial"], index=0)
        font_size = st.selectbox("Font Size", ["Small", "Medium", "Large"], index=1)
        
        set_state('font_family', font_family)
        set_state('font_size', font_size)

def render_model_settings():
    """Render model configuration settings"""
    
    st.markdown("### Model Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Default Model Configuration")
        
        # Default model selection
        model_options = ["EfficientNet-B0", "ResNet-50", "MobileNet-V2", "Vision Transformer"]
        default_model = st.selectbox("Default Model", model_options, index=0)
        set_state('default_model', default_model)
        
        # Model format preferences
        preferred_format = st.selectbox("Preferred Format", ["PyTorch", "ONNX", "TensorRT"], index=0)
        set_state('preferred_format', preferred_format)
        
        # Batch processing
        default_batch_size = st.number_input("Default Batch Size", min_value=1, max_value=64, value=8)
        set_state('default_batch_size', default_batch_size)
        
        # Confidence thresholds
        st.markdown("#### Confidence Thresholds")
        
        confidence_threshold = st.slider("Default Confidence Threshold", 0.0, 1.0, 0.7, 0.01)
        low_confidence_warning = st.slider("Low Confidence Warning", 0.0, 1.0, 0.5, 0.01)
        
        set_state('confidence_threshold', confidence_threshold)
        set_state('low_confidence_warning', low_confidence_warning)
    
    with col2:
        st.markdown("#### Model Caching")
        
        # Cache settings
        enable_model_cache = st.checkbox("Enable Model Caching", value=get_state('enable_model_cache', True))
        cache_size_gb = st.number_input("Cache Size (GB)", min_value=1, max_value=20, value=5)
        auto_cleanup = st.checkbox("Auto Cleanup Old Cache", value=get_state('auto_cleanup', True))
        
        set_state('enable_model_cache', enable_model_cache)
        set_state('cache_size_gb', cache_size_gb)
        set_state('auto_cleanup', auto_cleanup)
        
        # Performance settings
        st.markdown("#### Performance Settings")
        
        use_gpu = st.checkbox("Use GPU Acceleration", value=get_state('use_gpu', True))
        mixed_precision = st.checkbox("Enable Mixed Precision", value=get_state('mixed_precision', False))
        optimization_level = st.selectbox("Optimization Level", ["None", "Basic", "Aggressive"], index=1)
        
        set_state('use_gpu', use_gpu)
        set_state('mixed_precision', mixed_precision)
        set_state('optimization_level', optimization_level)
        
        # Model registry
        st.markdown("#### Model Registry")
        
        auto_register_models = st.checkbox("Auto Register New Models", value=get_state('auto_register_models', True))
        model_versioning = st.checkbox("Enable Model Versioning", value=get_state('model_versioning', True))
        
        set_state('auto_register_models', auto_register_models)
        set_state('model_versioning', model_versioning)

def render_data_settings():
    """Render data configuration settings"""
    
    st.markdown("### Data Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Dataset Configuration")
        
        # Default paths
        default_data_path = st.text_input("Default Data Path", value=get_state('default_data_path', 'data/'))
        manifest_path = st.text_input("Manifest Path", value=get_state('manifest_path', 'features/'))
        
        set_state('default_data_path', default_data_path)
        set_state('manifest_path', manifest_path)
        
        # Data preprocessing
        st.markdown("#### Preprocessing Options")
        
        auto_resize = st.checkbox("Auto Resize Images", value=get_state('auto_resize', True))
        target_size = st.selectbox("Target Image Size", ["224x224", "256x256", "512x512"], index=0)
        normalize_images = st.checkbox("Normalize Images", value=get_state('normalize_images', True))
        
        set_state('auto_resize', auto_resize)
        set_state('target_size', target_size)
        set_state('normalize_images', normalize_images)
        
        # Augmentation settings
        st.markdown("#### Data Augmentation")
        
        enable_augmentation = st.checkbox("Enable Augmentation", value=get_state('enable_augmentation', False))
        augmentation_strength = st.slider("Augmentation Strength", 0.0, 1.0, 0.3, 0.1)
        
        set_state('enable_augmentation', enable_augmentation)
        set_state('augmentation_strength', augmentation_strength)
    
    with col2:
        st.markdown("#### Storage Settings")
        
        # Cache settings
        enable_data_cache = st.checkbox("Enable Data Caching", value=get_state('enable_data_cache', True))
        data_cache_size_gb = st.number_input("Data Cache Size (GB)", min_value=1, max_value=50, value=10)
        
        set_state('enable_data_cache', enable_data_cache)
        set_state('data_cache_size_gb', data_cache_size_gb)
        
        # Data validation
        st.markdown("#### Data Validation")
        
        validate_images = st.checkbox("Validate Images on Load", value=get_state('validate_images', True))
        check_corruption = st.checkbox("Check for Corruption", value=get_state('check_corruption', False))
        skip_invalid = st.checkbox("Skip Invalid Files", value=get_state('skip_invalid', True))
        
        set_state('validate_images', validate_images)
        set_state('check_corruption', check_corruption)
        set_state('skip_invalid', skip_invalid)
        
        # Export settings
        st.markdown("#### Export Settings")
        
        default_export_format = st.selectbox("Default Export Format", ["CSV", "JSON", "Excel", "Parquet"], index=0)
        include_metadata = st.checkbox("Include Metadata", value=get_state('include_metadata', True))
        compress_exports = st.checkbox("Compress Exports", value=get_state('compress_exports', False))
        
        set_state('default_export_format', default_export_format)
        set_state('include_metadata', include_metadata)
        set_state('compress_exports', compress_exports)

def render_system_settings():
    """Render system configuration settings"""
    
    st.markdown("### System Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Resource Management")
        
        # CPU settings
        max_cpu_threads = st.number_input("Max CPU Threads", min_value=1, max_value=32, value=8)
        cpu_priority = st.selectbox("CPU Priority", ["Low", "Normal", "High"], index=1)
        
        set_state('max_cpu_threads', max_cpu_threads)
        set_state('cpu_priority', cpu_priority)
        
        # Memory settings
        max_memory_gb = st.number_input("Max Memory Usage (GB)", min_value=1, max_value=64, value=8)
        memory_monitoring = st.checkbox("Enable Memory Monitoring", value=get_state('memory_monitoring', True))
        
        set_state('max_memory_gb', max_memory_gb)
        set_state('memory_monitoring', memory_monitoring)
        
        # GPU settings
        gpu_memory_fraction = st.slider("GPU Memory Fraction", 0.1, 1.0, 0.8, 0.1)
        allow_gpu_growth = st.checkbox("Allow GPU Memory Growth", value=get_state('allow_gpu_growth', True))
        
        set_state('gpu_memory_fraction', gpu_memory_fraction)
        set_state('allow_gpu_growth', allow_gpu_growth)
    
    with col2:
        st.markdown("#### Logging & Monitoring")
        
        # Logging settings
        log_level = st.selectbox("Log Level", ["DEBUG", "INFO", "WARNING", "ERROR"], index=1)
        save_logs = st.checkbox("Save Logs to File", value=get_state('save_logs', True))
        max_log_size_mb = st.number_input("Max Log Size (MB)", min_value=1, max_value=1000, value=100)
        
        set_state('log_level', log_level)
        set_state('save_logs', save_logs)
        set_state('max_log_size_mb', max_log_size_mb)
        
        # Monitoring settings
        enable_profiling = st.checkbox("Enable Performance Profiling", value=get_state('enable_profiling', False))
        monitoring_interval = st.number_input("Monitoring Interval (seconds)", min_value=1, max_value=60, value=5)
        
        set_state('enable_profiling', enable_profiling)
        set_state('monitoring_interval', monitoring_interval)
        
        # Notifications
        st.markdown("#### Notifications")
        
        enable_notifications = st.checkbox("Enable Notifications", value=get_state('enable_notifications', True))
        notify_on_completion = st.checkbox("Notify on Task Completion", value=get_state('notify_on_completion', True))
        notify_on_error = st.checkbox("Notify on Error", value=get_state('notify_on_error', True))
        
        set_state('enable_notifications', enable_notifications)
        set_state('notify_on_completion', notify_on_completion)
        set_state('notify_on_error', notify_on_error)

def render_advanced_settings():
    """Render advanced configuration settings"""
    
    st.markdown("### Advanced Settings")
    
    # Configuration management
    st.markdown("#### Configuration Management")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📥 Export Settings", type="primary"):
            export_settings()
    
    with col2:
        uploaded_file = st.file_uploader("Import Settings", type=['json'], key="settings_import")
        if uploaded_file is not None:
            import_settings(uploaded_file)
    
    with col3:
        if st.button("🔄 Reset to Defaults"):
            if st.button("Confirm Reset", type="primary"):
                reset_to_defaults()
    
    # Debug settings
    st.markdown("#### Debug Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        debug_mode = st.checkbox("Enable Debug Mode", value=get_state('debug_mode', False))
        verbose_logging = st.checkbox("Verbose Logging", value=get_state('verbose_logging', False))
        show_performance_metrics = st.checkbox("Show Performance Metrics", value=get_state('show_performance_metrics', False))
        
        set_state('debug_mode', debug_mode)
        set_state('verbose_logging', verbose_logging)
        set_state('show_performance_metrics', show_performance_metrics)
    
    with col2:
        enable_telemetry = st.checkbox("Enable Telemetry", value=get_state('enable_telemetry', False))
        crash_reporting = st.checkbox("Crash Reporting", value=get_state('crash_reporting', True))
        
        set_state('enable_telemetry', enable_telemetry)
        set_state('crash_reporting', crash_reporting)
    
    # Experimental features
    st.markdown("#### Experimental Features")
    
    st.warning("⚠️ Experimental features may be unstable")
    
    experimental_ui = st.checkbox("Experimental UI Components", value=get_state('experimental_ui', False))
    beta_features = st.checkbox("Enable Beta Features", value=get_state('beta_features', False))
    feature_preview = st.checkbox("Feature Preview Mode", value=get_state('feature_preview', False))
    
    set_state('experimental_ui', experimental_ui)
    set_state('beta_features', beta_features)
    set_state('feature_preview', feature_preview)
    
    # System information
    render_system_information()
    
    # Database settings
    st.markdown("#### Database Settings")
    
    database_path = st.text_input("Database Path", value=get_state('database_path', 'lazarus.db'))
    backup_frequency = st.selectbox("Backup Frequency", ["Never", "Daily", "Weekly", "Monthly"], index=2)
    auto_vacuum = st.checkbox("Auto Vacuum Database", value=get_state('auto_vacuum', True))
    
    set_state('database_path', database_path)
    set_state('backup_frequency', backup_frequency)
    set_state('auto_vacuum', auto_vacuum)

def render_system_information():
    """Render system information section"""
    
    st.markdown("#### System Information")
    
    # System specs
    import platform
    import psutil
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.text(f"OS: {platform.system()} {platform.release()}")
        st.text(f"Processor: {platform.processor() or 'Unknown'}")
        st.text(f"Architecture: {platform.architecture()[0]}")
        st.text(f"Python: {platform.python_version()}")
    
    with col2:
        st.text(f"CPU Cores: {psutil.cpu_count(logical=False)}")
        st.text(f"CPU Threads: {psutil.cpu_count(logical=True)}")
        st.text(f"Total RAM: {psutil.virtual_memory().total / 1024**3:.1f} GB")
        st.text(f"Available RAM: {psutil.virtual_memory().available / 1024**3:.1f} GB")
    
    # Application info
    st.markdown("#### Application Information")
    
    app_info = {
        "Version": "1.0.0-beta",
        "Build": "2024.01.15",
        "Git Commit": "a1b2c3d",
        "Config Path": os.path.expanduser("~/.lazarus/config.json"),
        "Data Path": get_state('default_data_path', 'data/'),
        "Cache Size": "2.3 GB",
        "Session ID": get_state('session_id', 'unknown')
    }
    
    for key, value in app_info.items():
        st.text(f"{key}: {value}")

def export_settings():
    """Export current settings to JSON"""
    
    # Collect all settings from session state
    settings = {}
    
    # Define setting categories
    setting_keys = [
        'selected_theme', 'primary_color', 'secondary_color', 'accent_color',
        'sidebar_width', 'header_style', 'card_style', 'enable_animations',
        'default_model', 'preferred_format', 'confidence_threshold',
        'enable_model_cache', 'cache_size_gb', 'use_gpu', 'mixed_precision',
        'default_data_path', 'auto_resize', 'target_size', 'normalize_images',
        'max_cpu_threads', 'max_memory_gb', 'gpu_memory_fraction',
        'log_level', 'save_logs', 'enable_profiling', 'debug_mode'
    ]
    
    for key in setting_keys:
        if key in st.session_state:
            settings[key] = st.session_state[key]
    
    # Add metadata
    settings['_export_timestamp'] = datetime.now().isoformat()
    settings['_export_version'] = "1.0.0"
    
    # Convert to JSON
    settings_json = json.dumps(settings, indent=2)
    
    # Offer download
    st.download_button(
        label="💾 Download Settings",
        data=settings_json,
        file_name=f"lazarus_settings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )
    
    st.success("Settings exported successfully!")

def import_settings(uploaded_file):
    """Import settings from JSON file"""
    
    try:
        settings = json.load(uploaded_file)
        
        # Validate settings structure
        if '_export_version' not in settings:
            st.warning("Settings file may be from an older version")
        
        # Apply settings to session state
        imported_count = 0
        for key, value in settings.items():
            if not key.startswith('_'):  # Skip metadata
                st.session_state[key] = value
                set_state(key, value)
                imported_count += 1
        
        add_decision_log_entry("settings_imported", {
            "timestamp": datetime.now().isoformat(),
            "settings_count": imported_count
        })
        
        st.success(f"Successfully imported {imported_count} settings!")
        st.info("Some settings may require a page refresh to take effect.")
        
    except json.JSONDecodeError:
        st.error("Invalid JSON file. Please check the file format.")
    except Exception as e:
        st.error(f"Error importing settings: {str(e)}")

def reset_to_defaults():
    """Reset all settings to default values"""
    
    # Define default values
    defaults = {
        'selected_theme': 'Dark Mission Control',
        'primary_color': '#00ff88',
        'secondary_color': '#ff6b35',
        'accent_color': '#4ecdc4',
        'sidebar_width': 'Normal',
        'header_style': 'Standard',
        'card_style': 'Rounded',
        'enable_animations': True,
        'default_model': 'EfficientNet-B0',
        'preferred_format': 'PyTorch',
        'confidence_threshold': 0.7,
        'enable_model_cache': True,
        'cache_size_gb': 5,
        'use_gpu': True,
        'mixed_precision': False,
        'default_data_path': 'data/',
        'auto_resize': True,
        'target_size': '224x224',
        'normalize_images': True,
        'max_cpu_threads': 8,
        'max_memory_gb': 8,
        'gpu_memory_fraction': 0.8,
        'log_level': 'INFO',
        'save_logs': True,
        'enable_profiling': False,
        'debug_mode': False
    }
    
    # Apply defaults
    for key, value in defaults.items():
        st.session_state[key] = value
        set_state(key, value)
    
    add_decision_log_entry("settings_reset", {
        "timestamp": datetime.now().isoformat()
    })
    
    st.success("Settings reset to defaults!")
    st.info("Page refresh recommended to apply all changes.")