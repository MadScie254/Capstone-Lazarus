#!/bin/bash
# Docker entrypoint script for CAPSTONE-LAZARUS

set -e

echo "🌱 Starting CAPSTONE-LAZARUS..."

# Function to wait for service
wait_for_service() {
    local host=$1
    local port=$2
    local service_name=$3
    
    echo "Waiting for $service_name to be ready..."
    while ! nc -z $host $port; do
        echo "  $service_name is unavailable - sleeping"
        sleep 2
    done
    echo "  $service_name is ready!"
}

# Function to check environment
check_environment() {
    echo "Checking environment..."
    
    # Check Python version
    python_version=$(python --version 2>&1)
    echo "  Python version: $python_version"
    
    # Check TensorFlow
    if python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')" 2>/dev/null; then
        echo "  TensorFlow is available"
        
        # Check GPU availability
        if python -c "import tensorflow as tf; print(f'GPU Available: {len(tf.config.list_physical_devices(\"GPU\")) > 0}')" 2>/dev/null; then
            echo "  GPU check completed"
        fi
    else
        echo "  Warning: TensorFlow not available"
    fi
    
    # Check required directories
    for dir in data models logs experiments; do
        if [ ! -d "/app/$dir" ]; then
            echo "  Creating directory: /app/$dir"
            mkdir -p "/app/$dir"
        fi
    done
}

# Function to setup application
setup_application() {
    echo "Setting up application..."
    
    # Initialize configuration if not exists
    if [ ! -f "/app/config/config.yaml" ]; then
        echo "  Creating default configuration..."
        cat > /app/config/config.yaml << EOF
# CAPSTONE-LAZARUS Configuration
project:
  name: "CAPSTONE-LAZARUS"
  version: "1.0.0"
  description: "Advanced Plant Disease Detection System"

data:
  batch_size: 32
  image_size: 224
  validation_split: 0.2
  augmentation: true

model:
  architecture: "EfficientNetB3"
  num_classes: 17
  use_pretrained: true
  dropout_rate: 0.3

training:
  epochs: 50
  learning_rate: 0.001
  optimizer: "adam"
  early_stopping: true
  save_best_only: true

paths:
  data_dir: "/app/data"
  models_dir: "/app/models"
  logs_dir: "/app/logs"
  experiments_dir: "/app/experiments"
EOF
    fi
    
    # Create class labels if not exists
    if [ ! -f "/app/config/class_labels.json" ]; then
        echo "  Creating class labels..."
        cat > /app/config/class_labels.json << EOF
{
    "0": "Corn_Cercospora_leaf_spot",
    "1": "Corn_Common_rust",
    "2": "Corn_healthy",
    "3": "Corn_Northern_Leaf_Blight",
    "4": "Potato_Early_blight",
    "5": "Potato_healthy",
    "6": "Potato_Late_blight",
    "7": "Tomato_Bacterial_spot",
    "8": "Tomato_Early_blight",
    "9": "Tomato_healthy",
    "10": "Tomato_Late_blight",
    "11": "Tomato_Leaf_Mold",
    "12": "Tomato_Septoria_leaf_spot",
    "13": "Tomato_Spider_mites",
    "14": "Tomato_Target_Spot",
    "15": "Tomato_mosaic_virus",
    "16": "Tomato_Yellow_Leaf_Curl_Virus"
}
EOF
    fi
    
    echo "  Application setup completed"
}

# Function to run migrations or setup database
setup_database() {
    if [ ! -z "$DATABASE_URL" ] || [ ! -z "$POSTGRES_HOST" ]; then
        echo "Database configuration detected..."
        
        # Wait for database if specified
        if [ ! -z "$POSTGRES_HOST" ]; then
            wait_for_service $POSTGRES_HOST ${POSTGRES_PORT:-5432} "PostgreSQL"
        fi
        
        # Run any database migrations here
        echo "  Database setup completed"
    fi
}

# Function to start services based on command
start_service() {
    local service=$1
    
    case $service in
        "streamlit")
            echo "Starting Streamlit app..."
            exec streamlit run app/streamlit_app/main.py \
                --server.port=8501 \
                --server.address=0.0.0.0 \
                --server.headless=true \
                --browser.gatherUsageStats=false
            ;;
        
        "fastapi")
            echo "Starting FastAPI service..."
            exec uvicorn app.fastapi.main:app \
                --host=0.0.0.0 \
                --port=8000 \
                --workers=${WORKERS:-1} \
                --log-level=info
            ;;
        
        "training")
            echo "Starting training service..."
            exec python scripts/train_model.py "$@"
            ;;
        
        "jupyter")
            echo "Starting Jupyter Lab..."
            exec jupyter lab \
                --ip=0.0.0.0 \
                --port=8888 \
                --no-browser \
                --allow-root \
                --NotebookApp.token="${JUPYTER_TOKEN:-capstone_token}" \
                --NotebookApp.allow_origin='*'
            ;;
        
        "bash"|"shell")
            echo "Starting interactive shell..."
            exec /bin/bash
            ;;
        
        *)
            echo "Starting custom command: $@"
            exec "$@"
            ;;
    esac
}

# Main execution
main() {
    # Check environment
    check_environment
    
    # Setup application
    setup_application
    
    # Setup database if needed
    setup_database
    
    # Wait for external services if specified
    if [ ! -z "$WAIT_FOR_REDIS" ]; then
        wait_for_service ${REDIS_HOST:-redis} ${REDIS_PORT:-6379} "Redis"
    fi
    
    if [ ! -z "$WAIT_FOR_POSTGRES" ]; then
        wait_for_service ${POSTGRES_HOST:-postgres} ${POSTGRES_PORT:-5432} "PostgreSQL"
    fi
    
    # Start the requested service
    echo "🚀 Launching service..."
    start_service "$@"
}

# Run main function with all arguments
main "$@"