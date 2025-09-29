"""Compatibility utilities that proxy to the Lazarus Console managers."""

from importlib import import_module
from typing import TYPE_CHECKING

_dataset_manager = import_module("app.lazarus_console.utils.dataset_manager")
_model_manager = import_module("app.lazarus_console.utils.model_manager")

DatasetManager = _dataset_manager.DatasetManager
ModelManager = _model_manager.ModelManager

__all__ = ["DatasetManager", "ModelManager"]

if TYPE_CHECKING:
    from app.lazarus_console.utils.dataset_manager import DatasetManager as DatasetManagerType
    from app.lazarus_console.utils.model_manager import ModelManager as ModelManagerType

    DatasetManager = DatasetManagerType
    ModelManager = ModelManagerType
