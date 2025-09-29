from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


class ModelManager:
    """Thin wrapper around the model registry for console tooling."""

    def __init__(self, project_root: Path | str) -> None:
        self.project_root = Path(project_root).resolve()
        self.models_dir = self.project_root / "models"
        self.registry_path = self.models_dir / "model_registry.json"
        self.model_registry: Dict[str, object] = self._load_registry()
        self.available_models: List[str] = self._extract_available_models()
        self.default_model_name: Optional[str] = self._select_default_model()

    def _load_registry(self) -> Dict[str, object]:
        if not self.registry_path.exists():
            return {}
        try:
            with self.registry_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except json.JSONDecodeError:
            data = {}
        if not isinstance(data, dict):
            return {}
        return data

    def _extract_available_models(self) -> List[str]:
        models_section = self.model_registry.get("models")
        available: List[str] = []
        if isinstance(models_section, dict):
            for name, info in models_section.items():
                if isinstance(info, dict) and info.get("status") in {"ready", "available"}:
                    available.append(name)
            return sorted(available)
        if isinstance(self.model_registry, dict):
            for key, value in self.model_registry.items():
                if isinstance(value, dict) and value.get("status") in {"ready", "available"}:
                    available.append(key)
        return sorted(available)

    def _select_default_model(self) -> Optional[str]:
        models_section = self.model_registry.get("models") if isinstance(self.model_registry, dict) else None
        if isinstance(models_section, dict):
            for name, info in models_section.items():
                if isinstance(info, dict) and info.get("status") in {"ready", "available"}:
                    return name
            if models_section:
                return next(iter(models_section))
        return self.available_models[0] if self.available_models else None

    def get_default_model(self) -> Optional[str]:
        return self.default_model_name

    def get_model_info(self, name: str) -> Optional[Dict[str, object]]:
        models_section = self.model_registry.get("models") if isinstance(self.model_registry, dict) else None
        if isinstance(models_section, dict):
            info = models_section.get(name)
            if isinstance(info, dict):
                return info
        return None

    def iter_models(self) -> Iterable[str]:
        return iter(self.available_models)

    def resolve_model_path(self, name: Optional[str] = None) -> Optional[Path]:
        target = name or self.default_model_name
        if target is None:
            return None
        info = self.get_model_info(target)
        if not info:
            return None
        backend = info.get("backend") if isinstance(info, dict) else None
        if backend == "torchvision":
            return None
        path_value = info.get("model_path") or info.get("keras_model_path")
        if not isinstance(path_value, str):
            return None
        candidate = Path(path_value)
        if not candidate.is_absolute():
            candidate = self.project_root / path_value
        return candidate if candidate.exists() else None

    def get_console_model_specs(self) -> Dict[str, Dict[str, Any]]:
        models_section = self.model_registry.get("models") if isinstance(self.model_registry, dict) else None
        if not isinstance(models_section, dict):
            return {}

        configs: Dict[str, Dict[str, Any]] = {}
        for name, info in models_section.items():
            if not isinstance(info, dict):
                continue
            if info.get("status") not in {"ready", "available"}:
                continue
            backend = info.get("backend")
            if backend != "torchvision":
                continue

            configs[name] = {
                "label": info.get("label", name.replace("_", " ").title()),
                "torchvision_constructor": info.get("torchvision_constructor", name),
                "weights_enum": info.get("weights_enum"),
                "input_size": int(info.get("input_size", 224)),
                "onnx_filename": info.get("onnx_filename", f"{name}.onnx"),
                "ensemble_default_weight": float(info.get("ensemble_default_weight", 1.0)),
                "description": info.get("description", ""),
            }

        return configs
