"""Smoke tests for master trainer pipeline."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.master_trainer import MasterTrainer


@pytest.fixture
def synthetic_dataset(tmp_path):
    """Create a tiny synthetic dataset for testing."""
    data_root = tmp_path / "Data"
    classes = ["healthy", "diseased"]
    
    for cls in classes:
        cls_dir = data_root / cls
        cls_dir.mkdir(parents=True)
        # Create 3 tiny images per class
        for i in range(3):
            img = Image.fromarray(
                np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            )
            img.save(cls_dir / f"sample_{i:02d}.jpg")
    
    yield data_root
    
    # Cleanup
    if data_root.exists():
        shutil.rmtree(data_root)


@pytest.fixture
def test_config(tmp_path):
    """Create minimal config for testing."""
    config_path = tmp_path / "config.yaml"
    config_content = """
seed: 42
dropout_rate: 0.3
num_workers: 0
pin_memory: false
use_augmentations: false
use_class_balancing: false
optimizer: adamw
scheduler: cosine
use_amp: false

training_suite:
  fast_test:
    epochs: 1
    sample_ratio: 1.0
    max_images_per_class: 3
  default:
    batch_size_floor: 1
    patience: 1

models:
  - name: efficientnet_b0
    backbone: efficientnet_b0
    image_size: 64
    batch_size: 2
    learning_rate: 0.001
    weight_decay: 0.0001
    phases:
      - type: head
        epochs: 1
        freeze_backbone: true
"""
    config_path.write_text(config_content, encoding="utf-8")
    return config_path


def test_smoke_training_pipeline(synthetic_dataset, test_config, tmp_path):
    """Test end-to-end training pipeline with synthetic data."""
    # Setup
    models_dir = tmp_path / "models"
    experiments_csv = tmp_path / "experiments.csv"
    
    # Monkeypatch paths for test isolation
    import src.master_trainer as mt_module
    original_models_root = mt_module.MODELS_ROOT
    original_experiments_index = mt_module.EXPERIMENTS_INDEX
    
    mt_module.MODELS_ROOT = models_dir
    mt_module.EXPERIMENTS_INDEX = experiments_csv
    
    try:
        # Run training
        trainer = MasterTrainer(
            config_path=test_config,
            data_root=synthetic_dataset,
        )
        
        results = trainer.run(fast_test=True)
        
        # Assertions
        assert len(results) == 1, "Should complete one model run"
        result = results[0]
        
        # Check run directory created
        run_dir = models_dir / result["run_id"]
        assert run_dir.exists(), f"Run directory not created: {run_dir}"
        
        # Check artifacts
        best_checkpoint = run_dir / "best.pth"
        assert best_checkpoint.exists(), "Best checkpoint not saved"
        
        metadata_file = run_dir / "run_metadata.json"
        assert metadata_file.exists(), "Metadata not written"
        
        with metadata_file.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
            assert "metrics" in metadata
            assert "phase_history" in metadata
        
        # Check experiments.csv
        assert experiments_csv.exists(), "experiments.csv not created"
        content = experiments_csv.read_text(encoding="utf-8")
        assert "run_id" in content
        assert result["run_id"] in content
        
        # Verify Grad-CAM folder
        gradcam_dir = run_dir / "gradcam"
        assert gradcam_dir.exists(), "Grad-CAM directory not created"
        
    finally:
        # Restore
        mt_module.MODELS_ROOT = original_models_root
        mt_module.EXPERIMENTS_INDEX = original_experiments_index


def test_smoke_inference_with_saved_model(synthetic_dataset, test_config, tmp_path):
    """Test that saved model can be loaded and used for inference."""
    import torch
    from src.model_factory_torch import get_model
    
    models_dir = tmp_path / "models"
    
    # Monkeypatch
    import src.master_trainer as mt_module
    original_models_root = mt_module.MODELS_ROOT
    mt_module.MODELS_ROOT = models_dir
    
    try:
        trainer = MasterTrainer(
            config_path=test_config,
            data_root=synthetic_dataset,
        )
        
        results = trainer.run(fast_test=True)
        run_dir = models_dir / results[0]["run_id"]
        checkpoint_path = run_dir / "best.pth"
        
        # Load model
        model = get_model("efficientnet_b0", num_classes=2, pretrained=False)
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        
        # Run inference on synthetic image
        test_img = Image.fromarray(
            np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        )
        
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
        tensor = transform(test_img).unsqueeze(0)
        
        with torch.no_grad():
            output = model(tensor)
            probs = torch.softmax(output, dim=1)
        
        assert probs.shape == (1, 2), "Output shape mismatch"
        assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-5), "Probabilities don't sum to 1"
        
    finally:
        mt_module.MODELS_ROOT = original_models_root
