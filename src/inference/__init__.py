"""
Inference __init__ for CAPSTONE-LAZARUS
======================================
"""

from .predictor import (
    Predictor,
    PredictionResult,
    create_predictor,
    load_and_predict
)

__all__ = [
    'Predictor',
    'PredictionResult',
    'create_predictor',
    'load_and_predict'
]