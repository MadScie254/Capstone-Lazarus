# Configuration settings for the Plant Disease Detection System

# UI Configuration
APP_TITLE = "🌱 Plant Disease Detection System"
PAGE_ICON = "🌱"
LAYOUT = "wide"

# Color Scheme
PRIMARY_COLOR = "#2E8B57"
SECONDARY_COLOR = "#228B22"
BACKGROUND_COLOR = "#f0f8f0"

# Feature Flags
ENABLE_BATCH_PROCESSING = True
ENABLE_VISUALIZATION = True
ENABLE_EXPLAINABILITY = True
ENABLE_ANALYTICS = True

# Model Configuration
MODEL_PATH = './inception_lazarus'
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 32

# Class Information
class_info = {
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': {
        'description': 'A fungal disease causing gray-brown spots on corn leaves',
        'treatment': 'Apply fungicide, improve air circulation, crop rotation'
    },
    'Corn_(maize)___Common_rust_': {
        'description': 'Fungal disease with rust-colored pustules on leaves',
        'treatment': 'Use resistant varieties, apply fungicide if severe'
    },
    'Corn_(maize)___healthy': {
        'description': 'Healthy corn plant with no visible diseases',
        'treatment': 'Continue regular care and monitoring'
    },
    'Corn_(maize)___Northern_Leaf_Blight': {
        'description': 'Fungal disease causing cigar-shaped lesions on leaves',
        'treatment': 'Crop rotation, resistant varieties, fungicide application'
    },
    'Potato___Early_blight': {
        'description': 'Fungal disease causing brown spots with target-like rings',
        'treatment': 'Remove affected foliage, improve air circulation, fungicide'
    },
    'Potato___healthy': {
        'description': 'Healthy potato plant with no visible diseases',
        'treatment': 'Continue regular care and monitoring'
    },
    'Potato___Late_blight': {
        'description': 'Devastating fungal disease causing dark lesions',
        'treatment': 'Immediate fungicide treatment, remove affected plants'
    },
    'Tomato___Bacterial_spot': {
        'description': 'Bacterial disease causing dark spots on leaves and fruit',
        'treatment': 'Copper-based bactericide, improve air circulation'
    },
    'Tomato___Early_blight': {
        'description': 'Fungal disease causing brown spots with concentric rings',
        'treatment': 'Remove affected foliage, fungicide application'
    },
    'Tomato___healthy': {
        'description': 'Healthy tomato plant with no visible diseases',
        'treatment': 'Continue regular care and monitoring'
    },
    'Tomato___Late_blight': {
        'description': 'Serious fungal disease causing dark lesions',
        'treatment': 'Immediate fungicide treatment, improve ventilation'
    },
    'Tomato___Leaf_Mold': {
        'description': 'Fungal disease causing yellow spots and fuzzy growth',
        'treatment': 'Improve air circulation, reduce humidity, fungicide'
    },
    'Tomato___Septoria_leaf_spot': {
        'description': 'Fungal disease causing small circular spots with dark borders',
        'treatment': 'Remove affected leaves, fungicide application'
    },
    'Tomato___Spider_mites Two-spotted_spider_mite': {
        'description': 'Pest infestation causing stippled leaves and webbing',
        'treatment': 'Increase humidity, predatory mites, miticide if severe'
    },
    'Tomato___Target_Spot': {
        'description': 'Fungal disease causing spots with target-like appearance',
        'treatment': 'Fungicide application, improve air circulation'
    },
    'Tomato___Tomato_mosaic_virus': {
        'description': 'Viral disease causing mosaic patterns on leaves',
        'treatment': 'Remove infected plants, control aphids, use resistant varieties'
    },
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
        'description': 'Viral disease causing yellowing and curling of leaves',
        'treatment': 'Remove infected plants, control whiteflies, resistant varieties'
    }
}figuration file for Plant Disease Detection System
"""

# Model configuration
MODEL_PATH = './inception_lazarus'
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 32

# UI Configuration
APP_TITLE = "🌱 Plant Disease Detection System"
PAGE_ICON = "🌱"

# Color scheme
COLORS = {
    'primary': '#2E8B57',
    'secondary': '#90EE90',
    'success': '#4CAF50',
    'warning': '#FF9800',
    'error': '#F44336',
    'info': '#17a2b8'
}

# Severity color mapping
SEVERITY_COLORS = {
    'None': '#4CAF50',
    'Mild': '#8BC34A',
    'Moderate': '#FF9800',
    'High': '#FF5722',
    'Very High': '#F44336'
}

# Plant care tips
GENERAL_CARE_TIPS = {
    'watering': "Water plants early morning or late evening to reduce evaporation",
    'spacing': "Maintain proper plant spacing for good air circulation",
    'inspection': "Regularly inspect plants for early signs of disease",
    'sanitation': "Keep garden tools clean and sanitized",
    'rotation': "Practice crop rotation to prevent disease buildup",
    'mulching': "Use organic mulch to retain moisture and suppress weeds"
}

# Disease prevention tips
PREVENTION_TIPS = {
    'fungal': "Avoid overhead watering, ensure good drainage, provide adequate spacing",
    'bacterial': "Use drip irrigation, sanitize tools, avoid working with wet plants",
    'viral': "Control insect vectors, remove infected plants immediately, use virus-free seeds"
}

# Emergency contacts (example)
EMERGENCY_CONTACTS = {
    'agricultural_extension': "Contact your local agricultural extension office",
    'plant_pathologist': "Consult with a certified plant pathologist",
    'pesticide_info': "National Pesticide Information Center: 1-800-858-7378"
}

# File upload settings
ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg']
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Performance thresholds
CONFIDENCE_THRESHOLD = 0.7
HIGH_RISK_DISEASES = ['Late Blight', 'Tomato Yellow Leaf Curl Virus', 'Bacterial Spot']

# Analytics settings
CHART_HEIGHT = 400
CHART_WIDTH = 600

# Cache settings
CACHE_TTL = 3600  # 1 hour

# Feature flags
FEATURES = {
    'batch_processing': True,
    'explainability': True,
    'model_analytics': True,
    'data_insights': True,
    'treatment_recommendations': True,
    'download_results': True
}

# API settings (for future expansion)
API_SETTINGS = {
    'rate_limit': 100,  # requests per minute
    'timeout': 30,  # seconds
    'retry_attempts': 3
}