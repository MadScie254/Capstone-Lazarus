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
}

# Class names list (for easy access)
class_names = list(class_info.keys())