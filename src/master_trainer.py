"""Master training orchestration for CAPSTONE-LAZARUS.

This module coordinates sequential model training runs using the PyTorch
training stack, manages artifacts, generates explainability assets, and
records experiment metadata for dashboard consumption.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import shutil
import subprocess
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)

from src.data_utils_torch import create_subset_loader, make_dataloaders
from src.model_factory_torch import PlantDiseaseModel, get_model
from src.training_torch import Trainer as TorchTrainer

try:  # Optional runtime telemetry
    import psutil
except ImportError:  # pragma: no cover - optional dependency
    psutil = None  # type: ignore[assignment]

LOGGER_NAME = "master_trainer"
LOG_FILE = Path("logs") / "ops.log"
DATA_ROOT_CANDIDATES = [Path("./Data"), Path("./data")]


def _resolve_data_root() -> Path:
    for candidate in DATA_ROOT_CANDIDATES:
        if candidate.exists():
            return candidate
    return DATA_ROOT_CANDIDATES[0]


DATA_ROOT = _resolve_data_root()
MODELS_ROOT = Path("models")
EXPERIMENTS_INDEX = Path("experiments.csv")
GRADCAM_DEFAULT_IMAGES = 6
MAX_OOM_RETRIES = 3

EXPERIMENT_COLUMNS: Sequence[str] = (
    "run_id",
    "timestamp_utc",
    "commit_hash",
    "model_name",
    "backbone",
    "framework",
    "input_size",
    "params_count",
    "epochs_trained",
    "train_samples",
    "val_samples",
    "batch_size",
    "lr",
    "val_accuracy",
    "val_macro_f1",
    "val_macro_recall",
    "best_checkpoint_path",
    "onnx_path",
    "tflite_path",
    "gradcam_folder",
    "notes",
)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


@dataclass
class PhaseConfig:
    """Configuration for a single training phase."""

    type: str
    epochs: int
    freeze_backbone: bool = False
    unfreeze_blocks: Optional[int] = None
    learning_rate: Optional[float] = None


@dataclass
class ModelRunSpec:
    """Structured representation of a model run configuration."""

    name: str
    backbone: str
    image_size: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    phases: List[PhaseConfig] = field(default_factory=list)
    notes: Optional[str] = None

    @classmethod
    def from_raw(cls, raw: Dict[str, Any]) -> "ModelRunSpec":
        phases = [PhaseConfig(**phase) for phase in raw.get("phases", [])]
        return cls(
            name=raw["name"],
            backbone=raw.get("backbone", raw["name"]),
            image_size=raw.get("image_size", 224),
            batch_size=raw.get("batch_size", 8),
            learning_rate=raw.get("learning_rate", 1e-3),
            weight_decay=raw.get("weight_decay", 1e-4),
            phases=phases,
            notes=raw.get("notes"),
        )


class MasterTrainer:
    """Co-ordinates multi-model training runs and artifact generation."""

    def __init__(
        self,
        config_path: Path = Path("config.yaml"),
        models_list_path: Path = Path("config/models_list.json"),
        data_root: Optional[Path] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.project_root = Path.cwd()
        self.config_path = config_path
        self.models_list_path = models_list_path
        self.data_root = data_root if data_root is not None else _resolve_data_root()
        self.logger = logger or self._configure_logger()
        self.global_config = self._load_global_config()
        self.training_suite = self.global_config.get("training_suite", {})
        self.fast_test_cfg = (self.training_suite.get("fast_test") or {}).copy()
        self.default_suite_cfg = (self.training_suite.get("default") or {}).copy()
        self.models_raw = self._load_models_catalog()
        self.models: List[ModelRunSpec] = [ModelRunSpec.from_raw(m) for m in self.models_raw]

        self.logger.debug("Loaded %d model specifications", len(self.models))

        if not self.data_root.exists():
            message = f"Dataset folder not found. Please place images under './Data' or './data'. (looked for: {self.data_root})"
            self.logger.error(message)
            raise FileNotFoundError(message)

        MODELS_ROOT.mkdir(parents=True, exist_ok=True)
        EXPERIMENTS_INDEX.parent.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------------------------------------------
    def run(
        self,
        model_names: Optional[Sequence[str]] = None,
        fast_test: bool = False,
    ) -> List[Dict[str, Any]]:
        """Execute one or more model runs sequentially."""

        selected = [spec for spec in self.models if (not model_names or spec.name in model_names)]
        if not selected:
            raise ValueError("No matching models found for requested execution")

        results: List[Dict[str, Any]] = []
        for spec in selected:
            self.logger.info("Starting run for model '%s' (fast_test=%s)", spec.name, fast_test)
            result = self._run_single_model(spec, fast_test=fast_test)
            results.append(result)
        return results

    def smoke_test(self) -> Dict[str, Any]:
        """Convenience helper for automated smoke tests."""
        first_model = self.models[0]
        return self._run_single_model(first_model, fast_test=True)

    # --------------------------------------------------------------------------------------------------
    # Internal helpers
    # --------------------------------------------------------------------------------------------------
    def _configure_logger(self) -> logging.Logger:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger(LOGGER_NAME)
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            formatter = logging.Formatter(
                fmt="%(asctime)s | %(levelname)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
            file_handler.setFormatter(formatter)
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
        return logger

    def _load_global_config(self) -> Dict[str, Any]:
        import yaml

        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with self.config_path.open("r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        return cfg or {}

    def _load_models_catalog(self) -> List[Dict[str, Any]]:
        models: List[Dict[str, Any]] = self.global_config.get("models", [])
        if self.models_list_path.exists():
            with self.models_list_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            models.extend(data.get("models", []))
        dedup: Dict[str, Dict[str, Any]] = {}
        for model in models:
            dedup[model["name"]] = model
        return list(dedup.values())

    def _run_single_model(self, spec: ModelRunSpec, fast_test: bool) -> Dict[str, Any]:
        run_id = self._build_run_id(spec.name)
        commit_hash = self._current_commit_hash()
        run_dir = MODELS_ROOT / run_id
        checkpoints_dir = run_dir / "checkpoints"
        gradcam_dir = run_dir / "gradcam"
        exports_dir = run_dir

        for path in (run_dir, checkpoints_dir, gradcam_dir):
            path.mkdir(parents=True, exist_ok=True)

        base_seed = int(self.global_config.get("seed", 42))
        self._set_global_seeds(base_seed + abs(hash(spec.name)) % 1000)

        active_batch_size = spec.batch_size
        batch_size_floor = max(1, self.default_suite_cfg.get("batch_size_floor", 2))
        oom_retries = 0
        forced_cpu = False

        # Prepare dataloaders (may be recreated on fallback)
        loaders = self._prepare_dataloaders(spec, active_batch_size, fast_test)
        train_loader, val_loader = loaders

        # Instantiate model
        model = get_model(
            backbone=spec.backbone,
            num_classes=self._infer_num_classes(train_loader),
            pretrained=True,
            dropout_rate=self.global_config.get("dropout_rate", 0.3),
        )

        best_state = deepcopy(model.state_dict())
        best_metrics = {"val_accuracy": 0.0, "val_macro_f1": 0.0, "val_macro_recall": 0.0}
        total_epochs_trained = 0
        phase_history: List[Dict[str, Any]] = []

        # Determine training phases
        phases = list(spec.phases)
        if fast_test:
            phases = [PhaseConfig(type="head", epochs=self.fast_test_cfg.get("epochs", 1), freeze_backbone=True)]

        for phase in phases:
            self._configure_phase(model, phase, spec, fast_test)
            phase_cfg = self._build_phase_training_config(spec, phase, active_batch_size)

            while True:
                try:
                    trainer = TorchTrainer(
                        model=model,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        config=phase_cfg,
                        save_dir=str(checkpoints_dir),
                        device_override="cpu" if forced_cpu else None,
                    )
                    history = trainer.train(epochs=phase_cfg.get("epochs"))
                    total_epochs_trained += len(history.get("train_loss", []))

                    metrics, predictions, labels = self._evaluate_model(trainer.model, val_loader, trainer.device)
                    self.logger.info(
                        "Phase '%s' complete | Acc=%.4f | F1=%.4f", phase.type, metrics["val_accuracy"], metrics["val_macro_f1"]
                    )

                    if metrics["val_accuracy"] >= best_metrics.get("val_accuracy", 0):
                        best_state = deepcopy(trainer.model.state_dict())
                        best_metrics = metrics

                    phase_history.append(
                        {
                            "phase": phase.type,
                            "epochs": phase_cfg.get("epochs"),
                            "metrics": metrics,
                            "batch_size": active_batch_size,
                        }
                    )
                    break

                except RuntimeError as exc:
                    if "out of memory" not in str(exc).lower():
                        raise

                    self.logger.warning("CUDA OOM encountered during phase '%s': %s", phase.type, exc)
                    torch.cuda.empty_cache()
                    oom_retries += 1
                    if oom_retries >= MAX_OOM_RETRIES and not forced_cpu and torch.cuda.is_available():
                        self.logger.warning("Switching to CPU execution after repeated OOM events")
                        forced_cpu = True
                        continue

                    new_batch = max(active_batch_size // 2, batch_size_floor)
                    if new_batch == active_batch_size:
                        self.logger.error("Unable to reduce batch size further; aborting run")
                        raise
                    active_batch_size = new_batch
                    self.logger.info("Retrying with reduced batch size %d", active_batch_size)
                    loaders = self._prepare_dataloaders(spec, active_batch_size, fast_test)
                    train_loader, val_loader = loaders

        # Restore best weights for export/evaluation
        model.load_state_dict(best_state)
        device = torch.device("cuda" if torch.cuda.is_available() and not forced_cpu else "cpu")
        model.to(device)

        metrics, predictions, labels = self._evaluate_model(model, val_loader, device)
        confusion = confusion_matrix(labels, predictions)

        class_names = self._extract_class_names(train_loader)
        report_paths = self._generate_reports(
            run_dir=run_dir,
            metrics=metrics,
            confusion_matrix_values=confusion,
            class_names=class_names,
        )

        onnx_path, tflite_path, export_errors = self._export_models(
            model=model,
            run_dir=exports_dir,
            input_size=spec.image_size,
            device=device,
        )

        gradcam_folder = gradcam_dir
        self._generate_gradcam_gallery(
            model=model,
            val_loader=val_loader,
            output_dir=gradcam_folder,
            device=device,
        )

        best_checkpoint_path = self._save_best_checkpoint(model, run_dir)
        self._write_run_metadata(
            run_dir=run_dir,
            spec=spec,
            metrics=metrics,
            total_epochs=total_epochs_trained,
            phase_history=phase_history,
            batch_size=active_batch_size,
            forced_cpu=forced_cpu,
            export_errors=export_errors,
        )

        def _safe_relative(path: Path) -> str:
            try:
                return str(path.relative_to(self.project_root))
            except ValueError:
                return str(path)

        experiment_row = {
            "run_id": run_id,
            "timestamp_utc": datetime.utcnow().isoformat(),
            "commit_hash": commit_hash,
            "model_name": spec.name,
            "backbone": spec.backbone,
            "framework": "pytorch",
            "input_size": spec.image_size,
            "params_count": int(sum(p.numel() for p in model.parameters())),
            "epochs_trained": total_epochs_trained,
            "train_samples": len(train_loader.dataset),
            "val_samples": len(val_loader.dataset),
            "batch_size": active_batch_size,
            "lr": spec.learning_rate,
            "val_accuracy": metrics["val_accuracy"],
            "val_macro_f1": metrics["val_macro_f1"],
            "val_macro_recall": metrics["val_macro_recall"],
            "best_checkpoint_path": _safe_relative(best_checkpoint_path),
            "onnx_path": _safe_relative(onnx_path) if onnx_path else "",
            "tflite_path": _safe_relative(tflite_path) if tflite_path else "",
            "gradcam_folder": _safe_relative(gradcam_folder),
            "notes": spec.notes or "",
        }
        self._append_experiment_row(experiment_row)

        if export_errors:
            error_log = run_dir / "export_error.txt"
            with error_log.open("w", encoding="utf-8") as fh:
                for message in export_errors:
                    fh.write(f"{message}\n")

        return {
            "run_id": run_id,
            "metrics": metrics,
            "run_dir": str(run_dir),
            "epochs_trained": total_epochs_trained,
            "forced_cpu": forced_cpu,
            "phase_history": phase_history,
        }

    def _prepare_dataloaders(
        self,
        spec: ModelRunSpec,
        batch_size: int,
        fast_test: bool,
    ) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        loader_cfg = {
            "image_size": spec.image_size,
            "batch_size": batch_size,
            "num_workers": self.global_config.get("num_workers", 2),
            "pin_memory": self.global_config.get("pin_memory", False),
            "use_augmentations": self.global_config.get("use_augmentations", True),
            "augmentation_strength": self.global_config.get("augmentation_strength", "medium"),
            "seed": self.global_config.get("seed", 42),
            "use_class_balancing": self.global_config.get("use_class_balancing", True),
        }

        if fast_test:
            ratio = float(self.fast_test_cfg.get("sample_ratio", 0.02))
            max_per_class = int(self.fast_test_cfg.get("max_images_per_class", 2))
            subset_size = max_per_class * self._infer_num_classes_from_disk()
            dataset_size = self._count_dataset_samples()
            estimated_size = max(int(dataset_size * ratio), subset_size)
            subset_size = max(estimated_size, subset_size)
            train_loader = create_subset_loader(
                str(self.data_root), loader_cfg, subset_size=subset_size, split="train"
            )
            val_loader = create_subset_loader(
                str(self.data_root), loader_cfg, subset_size=max(4, subset_size // 2), split="val"
            )
            return train_loader, val_loader

        train_loader, val_loader = make_dataloaders(str(self.data_root), loader_cfg)
        return train_loader, val_loader

    def _configure_phase(
        self,
        model: PlantDiseaseModel,
        phase: PhaseConfig,
        spec: ModelRunSpec,
        fast_test: bool,
    ) -> None:
        if phase.type == "head" or phase.freeze_backbone:
            model.freeze_backbone()
        if phase.type == "finetune" and phase.unfreeze_blocks:
            model.unfreeze_last_n_layers(phase.unfreeze_blocks)
        if fast_test:
            # Ensure backbone stays frozen in fast mode
            model.freeze_backbone()

    def _build_phase_training_config(
        self,
        spec: ModelRunSpec,
        phase: PhaseConfig,
        batch_size: int,
    ) -> Dict[str, Any]:
        lr = phase.learning_rate or spec.learning_rate
        if phase.type == "finetune":
            lr *= self.default_suite_cfg.get("finetune", {}).get("lr_scale", 0.2)
        return {
            "epochs": phase.epochs,
            "learning_rate": lr,
            "weight_decay": spec.weight_decay,
            "optimizer": self.global_config.get("optimizer", "adamw"),
            "scheduler": self.global_config.get("scheduler", "cosine"),
            "batch_size": batch_size,
            "use_amp": self.global_config.get("use_amp", True),
            "early_stopping_patience": self.default_suite_cfg.get("patience", 3),
            "save_every": max(phase.epochs, 1),
        }

    def _evaluate_model(
        self,
        model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        device: torch.device,
    ) -> Tuple[Dict[str, float], List[int], List[int]]:
        model.eval()
        predictions: List[int] = []
        labels: List[int] = []

        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                predictions.extend(preds.cpu().tolist())
                labels.extend(targets.cpu().tolist())

        accuracy = accuracy_score(labels, predictions)
        macro_f1 = f1_score(labels, predictions, average="macro", zero_division=0)
        macro_recall = recall_score(labels, predictions, average="macro", zero_division=0)

        metrics = {
            "val_accuracy": float(accuracy),
            "val_macro_f1": float(macro_f1),
            "val_macro_recall": float(macro_recall),
        }
        return metrics, predictions, labels

    def _generate_reports(
        self,
        run_dir: Path,
        metrics: Dict[str, float],
        confusion_matrix_values: np.ndarray,
        class_names: List[str],
    ) -> Dict[str, Path]:
        import matplotlib.pyplot as plt
        import seaborn as sns

        cm_fig = run_dir / "confusion_matrix.png"
        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_matrix_values, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig(cm_fig, dpi=200)
        plt.close()

        report_path = run_dir / "report.html"
        report_html = [
            "<html><head><title>Training Report</title></head><body>",
            f"<h1>Run Report</h1>",
            f"<p><strong>Validation Accuracy:</strong> {metrics['val_accuracy']:.4f}</p>",
            f"<p><strong>Macro F1:</strong> {metrics['val_macro_f1']:.4f}</p>",
            f"<p><strong>Macro Recall:</strong> {metrics['val_macro_recall']:.4f}</p>",
            f"<img src='{cm_fig.name}' alt='Confusion Matrix' style='width:100%;max-width:960px;' />",
            "</body></html>",
        ]
        with report_path.open("w", encoding="utf-8") as fh:
            fh.write("\n".join(report_html))
        return {"confusion_matrix": cm_fig, "report": report_path}

    def _export_models(
        self,
        model: torch.nn.Module,
        run_dir: Path,
        input_size: int,
        device: torch.device,
    ) -> Tuple[Optional[Path], Optional[Path], List[str]]:
        export_errors: List[str] = []
        onnx_path = run_dir / "model.onnx"
        try:
            dummy = torch.randn(1, 3, input_size, input_size, device=device)
            torch.onnx.export(
                model,
                dummy,
                onnx_path,
                export_params=True,
                opset_version=12,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            )
        except Exception as exc:  # pragma: no cover - conversion may fail in CI
            export_errors.append(f"ONNX export failed: {exc}")
            onnx_path = None

        tflite_path: Optional[Path] = None
        try:
            import tensorflow as tf  # type: ignore

            if onnx_path and onnx_path.exists():
                import onnx
                from onnx_tf.backend import prepare  # type: ignore

                onnx_model = onnx.load(str(onnx_path))
                tf_rep = prepare(onnx_model)
                saved_model_dir = run_dir / "saved_model_temp"
                tf_rep.export_graph(str(saved_model_dir))
                converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                tflite_model = converter.convert()
                tflite_path = run_dir / "model.tflite"
                with tflite_path.open("wb") as fh:
                    fh.write(tflite_model)
                shutil.rmtree(saved_model_dir, ignore_errors=True)
        except Exception as exc:  # pragma: no cover - depends on optional deps
            export_errors.append(f"TFLite export failed: {exc}")
            tflite_path = None

        return onnx_path, tflite_path, export_errors

    def _generate_gradcam_gallery(
        self,
        model: torch.nn.Module,
        val_loader: torch.utils.data.DataLoader,
        output_dir: Path,
        device: torch.device,
    ) -> None:
        try:
            from pytorch_grad_cam import GradCAM
            from pytorch_grad_cam.utils.image import show_cam_on_image
            from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        except ImportError:  # pragma: no cover
            self.logger.warning("pytorch-grad-cam not available; skipping Grad-CAM generation")
            return

        target_layers = [layer for layer in model.backbone.modules() if isinstance(layer, torch.nn.Conv2d)]
        if not target_layers:
            self.logger.warning("No convolutional layers detected for Grad-CAM")
            return

        cam = GradCAM(model=model, target_layers=[target_layers[-1]], use_cuda=device.type == "cuda")
        saved = 0
        output_dir.mkdir(parents=True, exist_ok=True)

        for images, labels in val_loader:
            images = images.to(device)
            targets = [ClassifierOutputTarget(label.item()) for label in labels]
            grayscale_cams = cam(input_tensor=images, targets=targets)
            for idx in range(min(len(grayscale_cams), images.size(0))):
                cam_image = grayscale_cams[idx]
                tensor_image = images[idx].detach().cpu().numpy()
                tensor_image = self._denormalize_image(tensor_image)
                visualization = show_cam_on_image(tensor_image, cam_image, use_rgb=True)
                out_path = output_dir / f"gradcam_{saved:02d}.png"
                from PIL import Image

                Image.fromarray(visualization).save(out_path)
                saved += 1
                if saved >= GRADCAM_DEFAULT_IMAGES:
                    cam.activations_and_grads.release()
                    return
        cam.activations_and_grads.release()

    def _save_best_checkpoint(self, model: torch.nn.Module, run_dir: Path) -> Path:
        best_path = run_dir / "best.pth"
        torch.save(model.state_dict(), best_path)
        return best_path

    def _write_run_metadata(
        self,
        run_dir: Path,
        spec: ModelRunSpec,
        metrics: Dict[str, float],
        total_epochs: int,
        phase_history: List[Dict[str, Any]],
        batch_size: int,
        forced_cpu: bool,
        export_errors: List[str],
    ) -> None:
        metadata_path = run_dir / "run_metadata.json"
        hardware = self._collect_hardware_snapshot()
        metadata = {
            "model": asdict(spec),
            "metrics": metrics,
            "total_epochs": total_epochs,
            "phase_history": phase_history,
            "batch_size": batch_size,
            "forced_cpu": forced_cpu,
            "hardware": hardware,
            "export_errors": export_errors,
        }
        with metadata_path.open("w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2)

    def _append_experiment_row(self, row: Dict[str, Any]) -> None:
        temp_path = EXPERIMENTS_INDEX.with_suffix(".tmp")
        if EXPERIMENTS_INDEX.exists():
            existing = EXPERIMENTS_INDEX.read_text(encoding="utf-8")
        else:
            existing = ""

        with temp_path.open("w", newline="", encoding="utf-8") as tmp:
            writer = csv.DictWriter(tmp, fieldnames=EXPERIMENT_COLUMNS)
            if existing:
                tmp.write(existing)
                if not existing.endswith("\n"):
                    tmp.write("\n")
            else:
                writer.writeheader()
            writer.writerow(row)

        temp_path.replace(EXPERIMENTS_INDEX)

    def _build_run_id(self, model_name: str) -> str:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        commit_short = self._current_commit_hash()[:7]
        safe_name = model_name.replace(" ", "_")
        return f"{timestamp}_{safe_name}_{commit_short}"

    def _current_commit_hash(self) -> str:
        try:
            return (
                subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=self.project_root)
                .decode("utf-8")
                .strip()
            )
        except Exception:  # pragma: no cover - git not available
            return "unknown"

    def _infer_num_classes(self, loader: torch.utils.data.DataLoader) -> int:
        dataset = loader.dataset
        if hasattr(dataset, "dataset") and hasattr(dataset.dataset, "classes"):
            return len(dataset.dataset.classes)
        if hasattr(dataset, "classes"):
            return len(dataset.classes)
        raise ValueError("Unable to infer number of classes from dataset")

    def _infer_num_classes_from_disk(self) -> int:
        class_dirs = [p for p in self.data_root.iterdir() if p.is_dir()]
        return len(class_dirs)

    def _count_dataset_samples(self) -> int:
        try:
            from torchvision.datasets import ImageFolder

            dataset = ImageFolder(str(self.data_root))
            return len(dataset.samples)
        except Exception:  # pragma: no cover - fallback when torchvision unavailable
            total = 0
            for cls_dir in self.data_root.iterdir():
                if not cls_dir.is_dir():
                    continue
                for file in cls_dir.rglob("*"):
                    if file.is_file():
                        total += 1
            return total

    def _extract_class_names(self, loader: torch.utils.data.DataLoader) -> List[str]:
        dataset = loader.dataset
        if hasattr(dataset, "dataset") and hasattr(dataset.dataset, "classes"):
            return list(dataset.dataset.classes)
        if hasattr(dataset, "classes"):
            return list(dataset.classes)
        return [str(i) for i in range(self._infer_num_classes(loader))]

    def _denormalize_image(self, tensor: np.ndarray) -> np.ndarray:
        tensor = tensor.transpose(1, 2, 0)
        tensor = (tensor * IMAGENET_STD) + IMAGENET_MEAN
        tensor = np.clip(tensor, 0, 1)
        return tensor

    def _collect_hardware_snapshot(self) -> Dict[str, Any]:
        snapshot: Dict[str, Any] = {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        }
        if torch.cuda.is_available():
            snapshot["cuda_device"] = torch.cuda.get_device_name(0)
            snapshot["cuda_total_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
        if psutil:
            snapshot["memory_gb"] = psutil.virtual_memory().total / 1e9
            snapshot["cpu_count"] = psutil.cpu_count()
        return snapshot

    def _set_global_seeds(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def parse_args(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Master trainer for CAPSTONE-LAZARUS")
    parser.add_argument("--models", nargs="*", help="Subset of model names to run")
    parser.add_argument("--fast-test", action="store_true", help="Run in fast smoke-test mode")
    return parser.parse_args(args=args)


def main(cli_args: Optional[Sequence[str]] = None) -> None:
    args = parse_args(cli_args)
    trainer = MasterTrainer()
    trainer.run(model_names=args.models, fast_test=args.fast_test)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
