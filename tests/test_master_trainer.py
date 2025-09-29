from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
import pytest
import yaml

from src import master_trainer as mt


class TinyBackboneModel(torch.nn.Module):
    """Minimal convolutional network with the interfaces MasterTrainer expects."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.backbone = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(8, 16, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.classifier = torch.nn.Linear(16, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.backbone(x)
        pooled = self.pool(features)
        flattened = pooled.view(pooled.size(0), -1)
        return self.classifier(flattened)

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True

    def unfreeze_last_n_layers(self, _n: int) -> None:
        # For this tiny network we treat full backbone as a single block.
        self.unfreeze_backbone()


class DummyTrainer:
    """Lightweight stand-in for the real Trainer to keep tests fast."""

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: Any,
        val_loader: Any,
        config: dict[str, Any],
        save_dir: str,
        device_override: str | None = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.save_dir = save_dir
        self.device = torch.device(device_override or "cpu")

    def train(self, epochs: int | None = None) -> dict[str, list[float]]:
        steps = max(1, epochs or 1)
        return {"train_loss": [0.5] * steps, "val_loss": [0.6] * steps}


@pytest.fixture()
def tiny_dataset(tmp_path: Path) -> Path:
    """Create a tiny ImageFolder-compatible dataset."""

    data_root = tmp_path / "dataset"
    data_root.mkdir()
    rng = np.random.default_rng(42)
    for class_name in ("class_a", "class_b"):
        class_dir = data_root / class_name
        class_dir.mkdir()
        for index in range(6):
            image = (rng.random((64, 64, 3)) * 255).astype(np.uint8)
            Image.fromarray(image).save(class_dir / f"sample_{index}.jpg")
    return data_root


@pytest.fixture()
def tiny_config(tmp_path: Path) -> Path:
    """Write a minimal config file tailored for the master trainer unit test."""

    config = {
        "seed": 7,
        "use_augmentations": False,
        "training_suite": {
            "fast_test": {
                "enabled": True,
                "sample_ratio": 1.0,
                "max_images_per_class": 8,
                "epochs": 1,
            },
            "default": {
                "freeze_backbone": True,
                "head_training_epochs": 1,
                "batch_size_floor": 1,
                "patience": 1,
            },
        },
        "models": [
            {
                "name": "tiny_cnn",
                "backbone": "tiny",
                "image_size": 64,
                "batch_size": 2,
                "learning_rate": 0.001,
                "weight_decay": 0.0,
                "phases": [
                    {
                        "type": "head",
                        "epochs": 1,
                        "freeze_backbone": True,
                    }
                ],
            }
        ],
    }

    config_path = tmp_path / "config.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle)
    return config_path


def test_master_trainer_fast_run(monkeypatch: pytest.MonkeyPatch, tiny_dataset: Path, tiny_config: Path, tmp_path: Path) -> None:
    """MasterTrainer should run a fast smoke pass and record an experiment entry."""

    experiments_index = tmp_path / "experiments_test.csv"
    models_root = tmp_path / "trained_models"
    log_file = tmp_path / "ops.log"

    monkeypatch.setattr(mt, "EXPERIMENTS_INDEX", experiments_index)
    monkeypatch.setattr(mt, "MODELS_ROOT", models_root)
    monkeypatch.setattr(mt, "LOG_FILE", log_file)
    monkeypatch.setattr(mt, "TorchTrainer", DummyTrainer)
    monkeypatch.setattr(mt, "get_model", lambda **kwargs: TinyBackboneModel(kwargs["num_classes"]))

    trainer = mt.MasterTrainer(config_path=tiny_config, models_list_path=Path("nonexistent.json"), data_root=tiny_dataset)
    results = trainer.run(model_names=["tiny_cnn"], fast_test=True)
    assert len(results) == 1

    result = results[0]
    run_dir = Path(result["run_dir"])
    assert run_dir.exists()
    assert (run_dir / "run_metadata.json").exists()
    assert "val_accuracy" in result["metrics"]
    assert experiments_index.exists()

    with (run_dir / "run_metadata.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    assert metadata["model"]["name"] == "tiny_cnn"