"""
MLflow Logger stub for CAPSTONE-LAZARUS
=======================================

Provides a minimal interface around MLflow logging. If MLflow is not
installed or disabled in config, calls become no-ops.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from src.config import Config

logger = logging.getLogger(__name__)


class MLflowLogger:
    def __init__(self, config: Config):
        self.config = config
        self.enabled = bool(getattr(config.mlflow, 'enabled', False))
        self._mlflow = None
        if self.enabled:
            try:
                import mlflow
                mlflow.set_tracking_uri(config.mlflow.tracking_uri)
                mlflow.set_experiment(config.mlflow.experiment_name)
                self._mlflow = mlflow
                logger.info("MLflow initialized")
            except Exception as e:
                logger.warning(f"MLflow not available or failed to initialize: {e}")
                self.enabled = False

    def log_params(self, params: Dict[str, Any]):
        if self.enabled and self._mlflow:
            try:
                self._mlflow.log_params(params)
            except Exception as e:
                logger.warning(f"Failed to log params to MLflow: {e}")

    def log_metric(self, key: str, value: float, step: int | None = None):
        if self.enabled and self._mlflow:
            try:
                if step is None:
                    self._mlflow.log_metric(key, value)
                else:
                    self._mlflow.log_metric(key, value, step=step)
            except Exception as e:
                logger.warning(f"Failed to log metric to MLflow: {e}")

    def log_artifact(self, local_path: str, artifact_path: str | None = None):
        if self.enabled and self._mlflow:
            try:
                if artifact_path:
                    self._mlflow.log_artifact(local_path, artifact_path)
                else:
                    self._mlflow.log_artifact(local_path)
            except Exception as e:
                logger.warning(f"Failed to log artifact to MLflow: {e}")
