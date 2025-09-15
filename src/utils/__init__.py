"""
Utility modules __init__ for CAPSTONE-LAZARUS
============================================
"""

from .visualization import (
    plot_training_history,
    create_confusion_matrix,
    plot_class_distribution,
    plot_model_comparison,
    plot_feature_embeddings,
    plot_prediction_confidence,
    visualize_gradcam,
    plot_learning_curves,
    create_model_architecture_plot,
    save_multiple_plots,
    create_dashboard_layout
)

from .logging_utils import (
    setup_logging,
    log_execution_time,
    MetricsLogger,
    ExperimentLogger,
    get_logger,
    log_function_call,
    log_performance_metrics,
    setup_experiment_logging
)

__all__ = [
    # Visualization
    'plot_training_history',
    'create_confusion_matrix',
    'plot_class_distribution',
    'plot_model_comparison',
    'plot_feature_embeddings',
    'plot_prediction_confidence',
    'visualize_gradcam',
    'plot_learning_curves',
    'create_model_architecture_plot',
    'save_multiple_plots',
    'create_dashboard_layout',
    
    # Logging
    'setup_logging',
    'log_execution_time',
    'MetricsLogger',
    'ExperimentLogger',
    'get_logger',
    'log_function_call',
    'log_performance_metrics',
    'setup_experiment_logging'
]