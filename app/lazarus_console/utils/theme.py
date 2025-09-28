"""
Dark Professional Theme for Lazarus Console
Custom CSS and styling for immersive mission-control experience
"""

import streamlit as st

def apply_dark_theme():
    """Apply dark theme configuration"""
    
    # Configure Streamlit theme via config
    st.markdown("""
    <style>
    /* Import professional fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    /* Root variables for consistent theming */
    :root {
        --primary-bg: #0e1117;
        --secondary-bg: #1a1d29;
        --tertiary-bg: #262730;
        --accent-bg: #2d3748;
        
        --primary-text: #fafafa;
        --secondary-text: #e2e8f0;
        --muted-text: #94a3b8;
        --accent-text: #3182ce;
        
        --success: #10b981;
        --warning: #f59e0b;
        --error: #ef4444;
        --info: #3b82f6;
        
        --border-subtle: #374151;
        --border-strong: #4b5563;
        
        --font-primary: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        --font-mono: 'JetBrains Mono', 'Consolas', monospace;
        
        --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        
        --transition-fast: 150ms ease;
        --transition-normal: 250ms ease;
        --transition-slow: 350ms ease;
    }
    
    /* Global overrides */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: none;
    }
    
    /* Hide default Streamlit elements */
    .stDeployButton,
    .stDecoration,
    footer,
    header[data-testid="stHeader"] {
        display: none !important;
    }
    
    /* Main app background */
    .stApp {
        background: var(--primary-bg);
        font-family: var(--font-primary);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: var(--secondary-bg);
        border-right: 1px solid var(--border-subtle);
    }
    
    /* Typography improvements */
    h1, h2, h3, h4, h5, h6 {
        font-family: var(--font-primary);
        font-weight: 600;
        color: var(--primary-text);
        letter-spacing: -0.025em;
    }
    
    p, div, span {
        color: var(--secondary-text);
        font-family: var(--font-primary);
    }
    
    /* Code and metrics */
    code, pre {
        font-family: var(--font-mono);
        background: var(--tertiary-bg);
        border: 1px solid var(--border-subtle);
        border-radius: 6px;
        padding: 0.25rem 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

def inject_custom_css():
    """Inject custom component styles"""
    
    st.markdown("""
    <style>
    /* Mission Header Styles */
    .mission-header {
        background: linear-gradient(135deg, var(--secondary-bg) 0%, var(--tertiary-bg) 100%);
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 2rem;
        box-shadow: var(--shadow-lg);
    }
    
    .mission-title {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, var(--primary-text), var(--accent-text));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        line-height: 1.2;
    }
    
    .status-strip {
        display: flex;
        gap: 1.5rem;
        align-items: center;
        margin-top: 1rem;
        padding-top: 1rem;
        border-top: 1px solid var(--border-subtle);
    }
    
    .status-metric {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        background: var(--accent-bg);
        border-radius: 8px;
        border: 1px solid var(--border-subtle);
        transition: var(--transition-normal);
    }
    
    .status-metric:hover {
        border-color: var(--accent-text);
        box-shadow: 0 0 0 2px rgba(49, 130, 206, 0.1);
    }
    
    .status-value {
        font-weight: 600;
        font-family: var(--font-mono);
        color: var(--primary-text);
    }
    
    .status-label {
        font-size: 0.875rem;
        color: var(--muted-text);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Navigation Styles */
    .nav-container {
        display: flex;
        gap: 0.5rem;
        margin-top: 1.5rem;
        padding: 0.75rem;
        background: var(--accent-bg);
        border-radius: 10px;
        border: 1px solid var(--border-subtle);
    }
    
    .nav-button {
        flex: 1;
        padding: 0.75rem 1rem;
        background: transparent;
        border: 1px solid transparent;
        border-radius: 8px;
        color: var(--muted-text);
        font-weight: 500;
        text-align: center;
        cursor: pointer;
        transition: var(--transition-fast);
        text-decoration: none;
        font-size: 0.875rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .nav-button:hover {
        background: var(--tertiary-bg);
        color: var(--secondary-text);
        border-color: var(--border-strong);
    }
    
    .nav-button.active {
        background: var(--accent-text);
        color: white;
        border-color: var(--accent-text);
        box-shadow: var(--shadow-md);
    }
    
    /* Card Styles */
    .lazarus-card {
        background: var(--secondary-bg);
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        padding: 1.5rem;
        transition: var(--transition-normal);
        cursor: pointer;
        position: relative;
        overflow: hidden;
    }
    
    .lazarus-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, var(--accent-text), var(--success));
        opacity: 0;
        transition: var(--transition-normal);
    }
    
    .lazarus-card:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-lg);
        border-color: var(--border-strong);
    }
    
    .lazarus-card:hover::before {
        opacity: 1;
    }
    
    .card-title {
        font-size: 1.125rem;
        font-weight: 600;
        color: var(--primary-text);
        margin: 0 0 0.5rem 0;
    }
    
    .card-subtitle {
        font-size: 0.875rem;
        color: var(--muted-text);
        margin: 0 0 1rem 0;
    }
    
    .card-metric {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin: 0.5rem 0;
        padding: 0.5rem 0;
        border-top: 1px solid var(--border-subtle);
    }
    
    .card-metric:first-of-type {
        border-top: none;
        margin-top: 1rem;
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: var(--muted-text);
    }
    
    .metric-value {
        font-weight: 600;
        font-family: var(--font-mono);
        color: var(--primary-text);
    }
    
    /* Alert Styles */
    .alert {
        padding: 1rem 1.5rem;
        border-radius: 8px;
        border-left: 4px solid;
        margin: 1rem 0;
        position: relative;
    }
    
    .alert-success {
        background: rgba(16, 185, 129, 0.1);
        border-left-color: var(--success);
        color: var(--success);
    }
    
    .alert-warning {
        background: rgba(245, 158, 11, 0.1);
        border-left-color: var(--warning);
        color: var(--warning);
    }
    
    .alert-error {
        background: rgba(239, 68, 68, 0.1);
        border-left-color: var(--error);
        color: var(--error);
    }
    
    .alert-info {
        background: rgba(59, 130, 246, 0.1);
        border-left-color: var(--info);
        color: var(--info);
    }
    
    /* Toggle Switches */
    .toggle-container {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.5rem;
    }
    
    .toggle-switch {
        position: relative;
        width: 44px;
        height: 24px;
        background: var(--border-subtle);
        border-radius: 12px;
        cursor: pointer;
        transition: var(--transition-fast);
    }
    
    .toggle-switch.active {
        background: var(--accent-text);
    }
    
    .toggle-knob {
        position: absolute;
        top: 2px;
        left: 2px;
        width: 20px;
        height: 20px;
        background: white;
        border-radius: 10px;
        transition: var(--transition-fast);
        box-shadow: var(--shadow-sm);
    }
    
    .toggle-switch.active .toggle-knob {
        transform: translateX(20px);
    }
    
    /* Loading States */
    .skeleton {
        background: linear-gradient(90deg, var(--tertiary-bg) 25%, var(--accent-bg) 50%, var(--tertiary-bg) 75%);
        background-size: 200% 100%;
        animation: loading 1.5s infinite;
        border-radius: 4px;
    }
    
    @keyframes loading {
        0% { background-position: 200% 0; }
        100% { background-position: -200% 0; }
    }
    
    .pulse {
        animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .mission-header {
            padding: 1rem;
        }
        
        .mission-title {
            font-size: 1.5rem;
        }
        
        .status-strip {
            flex-direction: column;
            gap: 1rem;
        }
        
        .nav-container {
            flex-direction: column;
        }
        
        .nav-button {
            text-align: left;
        }
    }
    
    /* Custom Streamlit Component Overrides */
    .stSelectbox > div > div {
        background: var(--tertiary-bg);
        border: 1px solid var(--border-subtle);
        border-radius: 8px;
    }
    
    .stSlider > div > div {
        background: var(--tertiary-bg);
    }
    
    .stButton > button {
        background: var(--accent-text);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: var(--transition-fast);
    }
    
    .stButton > button:hover {
        background: #2c5aa0;
        transform: translateY(-1px);
        box-shadow: var(--shadow-md);
    }
    
    /* Progress bars */
    .stProgress > div > div {
        background: var(--accent-text);
        height: 8px;
        border-radius: 4px;
    }
    
    /* Metrics styling */
    .metric-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: var(--secondary-bg);
        border: 1px solid var(--border-subtle);
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        transition: var(--transition-fast);
    }
    
    .metric-card:hover {
        border-color: var(--accent-text);
        box-shadow: 0 0 0 2px rgba(49, 130, 206, 0.1);
    }
    
    .metric-card .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--primary-text);
        font-family: var(--font-mono);
    }
    
    .metric-card .metric-label {
        font-size: 0.875rem;
        color: var(--muted-text);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

def create_status_metric(label: str, value: str, status: str = "normal") -> str:
    """Create a status metric HTML component"""
    
    status_colors = {
        "normal": "var(--secondary-text)",
        "success": "var(--success)", 
        "warning": "var(--warning)",
        "error": "var(--error)",
        "info": "var(--info)"
    }
    
    color = status_colors.get(status, status_colors["normal"])
    
    return f"""
    <div class="status-metric">
        <div class="status-value" style="color: {color};">{value}</div>
        <div class="status-label">{label}</div>
    </div>
    """

def create_card(title: str, content: str, clickable: bool = True) -> str:
    """Create a card component"""
    
    cursor_style = "cursor: pointer;" if clickable else ""
    
    return f"""
    <div class="lazarus-card" style="{cursor_style}">
        <div class="card-title">{title}</div>
        <div class="card-content">{content}</div>
    </div>
    """

def create_alert(message: str, alert_type: str = "info") -> str:
    """Create an alert component"""
    
    return f"""
    <div class="alert alert-{alert_type}">
        {message}
    </div>
    """

def create_skeleton_loader(height: str = "20px", width: str = "100%") -> str:
    """Create a skeleton loading placeholder"""
    
    return f"""
    <div class="skeleton" style="height: {height}; width: {width};"></div>
    """