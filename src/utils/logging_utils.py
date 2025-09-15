"""
Logging utilities for CAPSTONE-LAZARUS
=====================================
"""

import logging
import logging.handlers
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import traceback
from contextlib import contextmanager
import time


class ColoredFormatter(logging.Formatter):
    """Colored log formatter for console output"""
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'ENDC': '\033[0m'        # End color
    }
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.COLORS['ENDC'])
        record.levelname = f"{log_color}{record.levelname}{self.COLORS['ENDC']}"
        return super().format(record)


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
                          'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                          'thread', 'threadName', 'processName', 'process', 'getMessage']:
                log_entry[key] = value
        
        return json.dumps(log_entry)


class MLflowLogHandler(logging.Handler):
    """Custom handler to send logs to MLflow"""
    
    def __init__(self, mlflow_client=None):
        super().__init__()
        self.mlflow_client = mlflow_client
    
    def emit(self, record):
        if self.mlflow_client:
            try:
                log_message = self.format(record)
                # Log to MLflow (implement based on your MLflow setup)
                # self.mlflow_client.log_text(log_message, "logs/training.log")
            except Exception:
                pass  # Don't break logging if MLflow fails


def setup_logging(log_level: str = "INFO",
                 log_dir: Optional[str] = None,
                 enable_console: bool = True,
                 enable_file: bool = True,
                 enable_json: bool = False,
                 max_file_size: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 5) -> logging.Logger:
    """
    Setup comprehensive logging configuration
    
    Args:
        log_level: Logging level
        log_dir: Directory for log files
        enable_console: Enable console logging
        enable_file: Enable file logging
        enable_json: Enable JSON formatted logging
        max_file_size: Maximum size of log files before rotation
        backup_count: Number of backup files to keep
        
    Returns:
        Configured logger
    """
    
    # Create main logger
    logger = logging.getLogger("capstone_lazarus")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers = []
    
    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = ColoredFormatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # File handlers
    if enable_file and log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        # Regular log file with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_path / "capstone_lazarus.log",
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        
        file_formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # Error log file
        error_handler = logging.handlers.RotatingFileHandler(
            filename=log_path / "errors.log",
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        logger.addHandler(error_handler)
        
        # JSON log file
        if enable_json:
            json_handler = logging.handlers.RotatingFileHandler(
                filename=log_path / "capstone_lazarus.json",
                maxBytes=max_file_size,
                backupCount=backup_count
            )
            json_handler.setFormatter(JSONFormatter())
            logger.addHandler(json_handler)
    
    return logger


@contextmanager
def log_execution_time(logger: logging.Logger, 
                      operation_name: str,
                      level: int = logging.INFO):
    """
    Context manager to log execution time
    
    Args:
        logger: Logger instance
        operation_name: Name of the operation
        level: Log level
    """
    start_time = time.time()
    logger.log(level, f"Starting {operation_name}")
    
    try:
        yield
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"{operation_name} failed after {execution_time:.2f}s: {e}")
        raise
    else:
        execution_time = time.time() - start_time
        logger.log(level, f"{operation_name} completed in {execution_time:.2f}s")


class MetricsLogger:
    """Logger for training and evaluation metrics"""
    
    def __init__(self, logger: logging.Logger, log_file: Optional[str] = None):
        self.logger = logger
        self.log_file = log_file
        self.metrics_history = []
    
    def log_metric(self, name: str, value: float, step: Optional[int] = None, 
                  timestamp: Optional[datetime] = None):
        """Log a single metric"""
        
        if timestamp is None:
            timestamp = datetime.now()
        
        metric_entry = {
            'name': name,
            'value': value,
            'step': step,
            'timestamp': timestamp.isoformat()
        }
        
        self.metrics_history.append(metric_entry)
        
        # Log to standard logger
        step_str = f" (step {step})" if step is not None else ""
        self.logger.info(f"Metric {name}: {value:.6f}{step_str}")
        
        # Write to metrics file
        if self.log_file:
            self._write_to_file(metric_entry)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log multiple metrics at once"""
        
        timestamp = datetime.now()
        
        for name, value in metrics.items():
            self.log_metric(name, value, step, timestamp)
    
    def log_hyperparameters(self, params: Dict[str, Any]):
        """Log hyperparameters"""
        
        self.logger.info("Hyperparameters:")
        for name, value in params.items():
            self.logger.info(f"  {name}: {value}")
        
        # Store hyperparameters
        hp_entry = {
            'type': 'hyperparameters',
            'params': params,
            'timestamp': datetime.now().isoformat()
        }
        
        if self.log_file:
            self._write_to_file(hp_entry)
    
    def log_model_summary(self, model_info: Dict[str, Any]):
        """Log model architecture summary"""
        
        self.logger.info("Model Summary:")
        for key, value in model_info.items():
            self.logger.info(f"  {key}: {value}")
        
        summary_entry = {
            'type': 'model_summary',
            'info': model_info,
            'timestamp': datetime.now().isoformat()
        }
        
        if self.log_file:
            self._write_to_file(summary_entry)
    
    def _write_to_file(self, entry: Dict[str, Any]):
        """Write entry to metrics file"""
        
        try:
            with open(self.log_file, 'a') as f:
                json.dump(entry, f)
                f.write('\n')
        except Exception as e:
            self.logger.error(f"Failed to write to metrics file: {e}")
    
    def get_metrics_history(self) -> List[Dict[str, Any]]:
        """Get complete metrics history"""
        return self.metrics_history.copy()
    
    def export_metrics(self, output_path: str, format: str = 'json'):
        """Export metrics to file"""
        
        output_path = Path(output_path)
        
        if format == 'json':
            with open(output_path, 'w') as f:
                json.dump(self.metrics_history, f, indent=2)
        
        elif format == 'csv':
            import pandas as pd
            
            # Convert to DataFrame
            df_data = []
            for entry in self.metrics_history:
                df_data.append({
                    'name': entry['name'],
                    'value': entry['value'],
                    'step': entry.get('step'),
                    'timestamp': entry['timestamp']
                })
            
            df = pd.DataFrame(df_data)
            df.to_csv(output_path, index=False)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        self.logger.info(f"Exported {len(self.metrics_history)} metrics to {output_path}")


class ExperimentLogger:
    """Logger for ML experiments"""
    
    def __init__(self, experiment_name: str, logger: logging.Logger, 
                 log_dir: Optional[str] = None):
        self.experiment_name = experiment_name
        self.logger = logger
        self.log_dir = Path(log_dir) if log_dir else None
        self.start_time = datetime.now()
        self.metrics_logger = None
        
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            metrics_file = self.log_dir / f"{experiment_name}_metrics.jsonl"
            self.metrics_logger = MetricsLogger(logger, str(metrics_file))
        else:
            self.metrics_logger = MetricsLogger(logger)
    
    def log_experiment_start(self, config: Dict[str, Any]):
        """Log experiment start with configuration"""
        
        self.logger.info(f"Starting experiment: {self.experiment_name}")
        self.logger.info(f"Start time: {self.start_time}")
        
        if self.metrics_logger:
            self.metrics_logger.log_hyperparameters(config)
        
        # Save config
        if self.log_dir:
            config_file = self.log_dir / f"{self.experiment_name}_config.json"
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2, default=str)
    
    def log_experiment_end(self, final_metrics: Optional[Dict[str, float]] = None):
        """Log experiment completion"""
        
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        self.logger.info(f"Experiment {self.experiment_name} completed")
        self.logger.info(f"Total duration: {duration}")
        
        if final_metrics and self.metrics_logger:
            self.logger.info("Final metrics:")
            self.metrics_logger.log_metrics(final_metrics)
        
        # Save experiment summary
        if self.log_dir:
            summary = {
                'experiment_name': self.experiment_name,
                'start_time': self.start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': duration.total_seconds(),
                'final_metrics': final_metrics or {}
            }
            
            summary_file = self.log_dir / f"{self.experiment_name}_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
    
    def log_checkpoint(self, epoch: int, metrics: Dict[str, float], 
                      model_path: Optional[str] = None):
        """Log training checkpoint"""
        
        self.logger.info(f"Checkpoint - Epoch {epoch}")
        self.metrics_logger.log_metrics(metrics, step=epoch)
        
        if model_path:
            self.logger.info(f"Model saved to: {model_path}")


# Utility functions
def get_logger(name: str = "capstone_lazarus") -> logging.Logger:
    """Get configured logger instance"""
    return logging.getLogger(name)


def log_function_call(logger: logging.Logger):
    """Decorator to log function calls"""
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
            
            try:
                result = func(*args, **kwargs)
                logger.debug(f"{func.__name__} completed successfully")
                return result
            except Exception as e:
                logger.error(f"{func.__name__} failed: {e}")
                raise
        
        return wrapper
    return decorator


def log_performance_metrics(logger: logging.Logger, 
                           metrics: Dict[str, float],
                           prefix: str = ""):
    """Log performance metrics in a formatted way"""
    
    if prefix:
        logger.info(f"{prefix} Performance Metrics:")
    else:
        logger.info("Performance Metrics:")
    
    for metric, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"  {metric}: {value:.6f}")
        else:
            logger.info(f"  {metric}: {value}")


def setup_experiment_logging(experiment_name: str,
                           log_dir: str,
                           log_level: str = "INFO") -> Tuple[logging.Logger, ExperimentLogger]:
    """
    Setup logging for an ML experiment
    
    Args:
        experiment_name: Name of the experiment
        log_dir: Directory for logs
        log_level: Logging level
        
    Returns:
        Tuple of (logger, experiment_logger)
    """
    
    # Setup main logging
    logger = setup_logging(
        log_level=log_level,
        log_dir=log_dir,
        enable_console=True,
        enable_file=True,
        enable_json=True
    )
    
    # Create experiment logger
    experiment_logger = ExperimentLogger(experiment_name, logger, log_dir)
    
    return logger, experiment_logger