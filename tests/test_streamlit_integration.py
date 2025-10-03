"""Smoke tests for Streamlit dashboard integration."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

import pytest


def test_streamlit_reads_experiments_csv(tmp_path, monkeypatch):
    """Test that dashboard can read and parse experiments.csv."""
    # Create a mock experiments.csv
    exp_csv = tmp_path / "experiments.csv"
    
    with exp_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "run_id", "timestamp_utc", "commit_hash", "model_name", "backbone",
            "framework", "input_size", "params_count", "epochs_trained",
            "train_samples", "val_samples", "batch_size", "lr",
            "val_accuracy", "val_macro_f1", "val_macro_recall",
            "best_checkpoint_path", "onnx_path", "tflite_path",
            "gradcam_folder", "notes"
        ])
        writer.writeheader()
        writer.writerow({
            "run_id": "20251004_120000_efficientnet_b0_abc1234",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "commit_hash": "abc1234567890",
            "model_name": "efficientnet_b0",
            "backbone": "efficientnet_b0",
            "framework": "pytorch",
            "input_size": 224,
            "params_count": 5300000,
            "epochs_trained": 5,
            "train_samples": 1000,
            "val_samples": 200,
            "batch_size": 8,
            "lr": 0.001,
            "val_accuracy": 0.932,
            "val_macro_f1": 0.918,
            "val_macro_recall": 0.925,
            "best_checkpoint_path": "models/run_001/best.pth",
            "onnx_path": "models/run_001/model.onnx",
            "tflite_path": "",
            "gradcam_folder": "models/run_001/gradcam",
            "notes": "Smoke test run"
        })
    
    # Monkeypatch the experiments path in the console module
    monkeypatch.chdir(tmp_path)
    
    import sys
    import importlib.util
    
    # Load the console module
    console_path = Path(__file__).parent.parent / "app" / "lazarus_console" / "__init__.py"
    if not console_path.exists():
        pytest.skip("Streamlit console module not found")
    
    spec = importlib.util.spec_from_file_location("lazarus_console", console_path)
    if spec is None or spec.loader is None:
        pytest.skip("Could not load console module")
    
    console_module = importlib.util.module_from_spec(spec)
    
    # Patch the experiments index path
    original_exp_path = getattr(console_module, "EXPERIMENTS_INDEX_PATH", None)
    console_module.EXPERIMENTS_INDEX_PATH = exp_csv
    
    try:
        spec.loader.exec_module(console_module)
        
        # Test load_experiments_index function
        if hasattr(console_module, "load_experiments_index"):
            df = console_module.load_experiments_index(limit=10)
            
            assert not df.empty, "Should load at least one experiment"
            assert len(df) == 1, "Should have exactly one row"
            assert df.iloc[0]["model_name"] == "efficientnet_b0"
            assert df.iloc[0]["val_macro_f1"] == pytest.approx(0.918, abs=1e-3)
        
        # Test build_home_metrics function
        if hasattr(console_module, "build_home_metrics"):
            metrics = console_module.build_home_metrics()
            
            assert "checkpoints" in metrics
            assert len(metrics["checkpoints"]) >= 1
            assert metrics["macro_f1"] == pytest.approx(0.918, abs=1e-3)
    
    finally:
        if original_exp_path is not None:
            console_module.EXPERIMENTS_INDEX_PATH = original_exp_path


def test_telemetry_logging():
    """Test that inference telemetry logging works."""
    from src.telemetry import log_inference, INFERENCE_LOG_PATH
    
    import tempfile
    import shutil
    
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Monkeypatch the log path
        import src.telemetry as telemetry_module
        original_log_path = telemetry_module.INFERENCE_LOG_PATH
        test_log_path = temp_dir / "logs" / "inference_log.csv"
        telemetry_module.INFERENCE_LOG_PATH = test_log_path
        
        # Log an inference
        log_inference(
            run_id="test_run_001",
            model_path="models/test/best.pth",
            image_name="test_image.jpg",
            top1_label="healthy",
            top1_confidence=0.95,
            latency_ms=42.5,
        )
        
        assert test_log_path.exists(), "Inference log not created"
        
        content = test_log_path.read_text(encoding="utf-8")
        assert "timestamp" in content
        assert "test_run_001" in content
        assert "healthy" in content
        assert "0.9500" in content
        
        # Log a second inference to test append
        log_inference(
            run_id="test_run_002",
            model_path="models/test/best.pth",
            image_name="test_image2.jpg",
            top1_label="diseased",
            top1_confidence=0.88,
            latency_ms=38.2,
        )
        
        lines = test_log_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 3, "Should have header + 2 data rows"
        
    finally:
        telemetry_module.INFERENCE_LOG_PATH = original_log_path
        shutil.rmtree(temp_dir, ignore_errors=True)
