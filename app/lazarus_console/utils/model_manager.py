from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional


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
        if isinstance(models_section, dict):
            return sorted(models_section.keys())
        if isinstance(self.model_registry, dict):
            return [key for key, value in self.model_registry.items() if isinstance(value, dict)]
        return []

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
        path_value = info.get("model_path") or info.get("keras_model_path")
        if not isinstance(path_value, str):
            return None
        candidate = Path(path_value)
        if not candidate.is_absolute():
            candidate = self.project_root / path_value
        return candidate if candidate.exists() else None
