"""
CAPSTONE-LAZARUS: Modern ML/DL Superintelligence Framework
=========================================================

A production-grade machine learning framework built with TensorFlow 2.20+ and Keras 3,
featuring automated hyperparameter tuning, neural architecture search, federated learning,
differential privacy, and comprehensive MLOps capabilities.

Architecture:
- config/: Configuration management and settings
- data/: ETL, validation, and dataset management
- models/: Model architectures, NAS, pruning, and quantization
- training/: Training loops, tuning, federated learning, and DP-SGD
- inference/: Prediction and serving capabilities
- experiments/: MLflow integration and experiment tracking
- utils/: Utilities for metrics, I/O, and explainability
- monitoring/: Drift detection and observability

Key Features:
- Neural Architecture Search (NAS) with weight sharing
- Federated learning with TensorFlow Federated
- Differential privacy with DP-SGD
- Automated hyperparameter optimization with Optuna
- Model compression via pruning and quantization
- Real-time monitoring and drift detection
- Interactive Streamlit playground
- Production-ready Docker deployment
"""

__version__ = "1.0.0"
__author__ = "CAPSTONE-LAZARUS Team"

import logging
import os
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

# Ensure critical directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

logger = logging.getLogger(__name__)
logger.info(f"CAPSTONE-LAZARUS v{__version__} initialized")
logger.info(f"Project root: {PROJECT_ROOT}")
logger.info(f"Data directory: {DATA_DIR}")
logger.info(f"Models directory: {MODELS_DIR}")