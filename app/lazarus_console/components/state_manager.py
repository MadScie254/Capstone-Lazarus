"""
Global State Management for Lazarus Console
Centralized session state management with persistent context across routes
"""

import streamlit as st
from typing import Any, Dict, Optional
import json
from pathlib import Path

# Default state configuration
DEFAULT_STATE = {
    # Navigation and UI
    'current_route': 'home',
    'previous_route': None,
    'sidebar_expanded': False,
    
    # Model and Dataset
    'selected_model': None,
    'available_models': [],
    'dataset_manifest': None,
    'dataset_version': 'v1.0',
    
    # Inference and Analysis
    'selected_images': [],
    'batch_results': None,
    'gradcam_target': None,
    'confidence_threshold': 0.5,
    'uncertainty_threshold': 0.3,
    
    # Performance and Hardware
    'amp_enabled': False,
    'onnx_enabled': False,
    'batch_size': 8,
    'precision': 'fp32',
    
    # Training and Monitoring
    'training_run_status': 'idle',
    'training_metrics': None,
    'checkpoint_timeline': [],
    
    # System Status
    'system_status': {
        'gpu_available': False,
        'vram_usage': 0,
        'cpu_usage': 0,
        'memory_usage': 0
    },
    
    # UI Preferences
    'theme': 'dark',
    'animations_enabled': True,
    'auto_refresh': True,
    'refresh_interval': 5.0,
    
    # Cache Management
    'cache_keys': {},
    'last_cache_clear': None,
    
    # Decision Log
    'decision_log': [],
    'deployment_gates': {
        'critical_recall_floor': 0.90,
        'macro_f1_floor': 0.85,
        'calibration_ece_ceiling': 0.15
    }
}

def initialize_global_state():
    """Initialize global session state with defaults"""
    
    # Initialize all default keys if not present
    for key, default_value in DEFAULT_STATE.items():
        if key not in st.session_state:
            st.session_state[key] = default_value
    
    # Load persisted config if exists
    load_persisted_config()
    
    # Initialize runtime state
    if 'app_start_time' not in st.session_state:
        st.session_state.app_start_time = st.session_state.get('app_start_time', None)
    
    if 'session_id' not in st.session_state:
        import uuid
        st.session_state.session_id = str(uuid.uuid4())[:8]

def get_state(key: str, default: Any = None) -> Any:
    """Get value from global state"""
    return st.session_state.get(key, default)

def set_state(key: str, value: Any) -> None:
    """Set value in global state"""
    st.session_state[key] = value

def update_state(updates: Dict[str, Any]) -> None:
    """Update multiple state values at once"""
    for key, value in updates.items():
        st.session_state[key] = value

def clear_cache_keys(pattern: str = None) -> None:
    """Clear cache keys matching pattern"""
    cache_keys = st.session_state.get('cache_keys', {})
    
    if pattern is None:
        # Clear all cache keys
        cache_keys.clear()
    else:
        # Clear keys matching pattern
        keys_to_remove = [k for k in cache_keys.keys() if pattern in k]
        for key in keys_to_remove:
            cache_keys.pop(key, None)
    
    st.session_state.cache_keys = cache_keys
    st.session_state.last_cache_clear = st.session_state.get('last_cache_clear', None)

def get_cache_key(model_name: str, precision: str = 'fp32', onnx: bool = False) -> str:
    """Generate cache key for model/session"""
    suffix = 'onnx' if onnx else 'pytorch'
    return f"{model_name}_{precision}_{suffix}"

def add_decision_log_entry(action: str, details: Dict[str, Any]) -> None:
    """Add entry to decision log"""
    entry = {
        'timestamp': st.session_state.get('timestamp', None),
        'session_id': st.session_state.get('session_id', 'unknown'),
        'action': action,
        'details': details,
        'state_snapshot': {
            'selected_model': get_state('selected_model'),
            'confidence_threshold': get_state('confidence_threshold'),
            'amp_enabled': get_state('amp_enabled'),
            'onnx_enabled': get_state('onnx_enabled')
        }
    }
    
    decision_log = st.session_state.get('decision_log', [])
    decision_log.append(entry)
    st.session_state.decision_log = decision_log[-100:]  # Keep last 100 entries

def save_persisted_config():
    """Save persistent configuration to disk"""
    try:
        config_dir = Path(__file__).parent.parent / 'assets'
        config_dir.mkdir(exist_ok=True)
        config_file = config_dir / 'user_config.json'
        
        # Only save user preferences, not runtime state
        persist_keys = [
            'theme', 'animations_enabled', 'auto_refresh', 'refresh_interval',
            'amp_enabled', 'onnx_enabled', 'batch_size', 'precision',
            'confidence_threshold', 'uncertainty_threshold', 'deployment_gates'
        ]
        
        config = {key: get_state(key) for key in persist_keys}
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
            
    except Exception as e:
        # Fail silently for config save errors
        pass

def load_persisted_config():
    """Load persistent configuration from disk"""
    try:
        config_file = Path(__file__).parent.parent / 'assets' / 'user_config.json'
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            # Update session state with loaded config
            for key, value in config.items():
                if key in DEFAULT_STATE:  # Only load known keys
                    st.session_state[key] = value
                    
    except Exception as e:
        # Fail silently for config load errors
        pass

def reset_state():
    """Reset all state to defaults"""
    for key in list(st.session_state.keys()):
        if key.startswith('_'):  # Don't reset Streamlit internals
            continue
        del st.session_state[key]
    
    initialize_global_state()

def get_system_status() -> Dict[str, Any]:
    """Get current system status"""
    import psutil
    import torch
    
    status = {
        'cpu_usage': psutil.cpu_percent(),
        'memory_usage': psutil.virtual_memory().percent,
        'gpu_available': torch.cuda.is_available()
    }
    
    if torch.cuda.is_available():
        status['gpu_name'] = torch.cuda.get_device_name(0)
        status['vram_total'] = torch.cuda.get_device_properties(0).total_memory / 1e9
        status['vram_usage'] = torch.cuda.memory_allocated() / 1e9
        status['vram_percent'] = (status['vram_usage'] / status['vram_total']) * 100
    
    return status

def update_system_status():
    """Update system status in global state"""
    try:
        status = get_system_status()
        st.session_state.system_status = status
    except Exception:
        # Fail silently if system monitoring unavailable
        pass