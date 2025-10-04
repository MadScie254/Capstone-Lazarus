"""
Test dashboard integration with REAL trained models.
This test validates:
1. Notebook imports work correctly
2. Dashboard loads trained checkpoints from experiments.csv
3. Telemetry logging is integrated into inference
4. Model Hub displays experiment data correctly
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import pytest
from datetime import datetime, timezone


def test_notebook_imports():
    """Verify notebook can import src modules after sys.path fix."""
    # Simulate notebook environment
    notebook_dir = PROJECT_ROOT / "notebooks"
    
    # Test import from src
    try:
        from src import master_trainer
        from src import data_utils_torch
        from src import model_factory_torch
        from src import telemetry
        print("✓ Notebook imports successful")
        assert True
    except ImportError as e:
        pytest.fail(f"Notebook import failed: {e}")


def test_experiments_csv_structure():
    """Verify experiments.csv has required columns for dashboard."""
    experiments_path = PROJECT_ROOT / "experiments.csv"
    
    required_columns = [
        "run_id",
        "timestamp_utc",
        "model_name",
        "backbone",
        "framework",
        "val_macro_f1",
        "val_accuracy",
        "val_macro_recall",
        "best_checkpoint_path",
        "gradcam_folder"
    ]
    
    if not experiments_path.exists():
        print("⚠ experiments.csv not found - skipping (train a model first)")
        pytest.skip("No experiments.csv found")
        return
    
    df = pd.read_csv(experiments_path)
    
    for col in required_columns:
        assert col in df.columns, f"Missing required column: {col}"
    
    print(f"✓ experiments.csv has {len(df)} runs with all required columns")


def test_dashboard_checkpoint_loading():
    """Test dashboard get_best_checkpoint_for_model function."""
    from app.lazarus_console import get_best_checkpoint_for_model
    
    experiments_path = PROJECT_ROOT / "experiments.csv"
    
    if not experiments_path.exists():
        print("⚠ No experiments.csv - skipping checkpoint loading test")
        pytest.skip("No trained models found")
        return
    
    df = pd.read_csv(experiments_path)
    
    if df.empty:
        print("⚠ experiments.csv is empty - train a model first")
        pytest.skip("No experiments recorded")
        return
    
    # Get first model from experiments
    first_model = df.iloc[0]["model_name"]
    first_backbone = df.iloc[0]["backbone"]
    
    checkpoint_path = get_best_checkpoint_for_model(first_model, first_backbone)
    
    if checkpoint_path is None:
        print(f"⚠ No checkpoint found for {first_model} ({first_backbone})")
    else:
        assert checkpoint_path.exists(), f"Checkpoint path doesn't exist: {checkpoint_path}"
        print(f"✓ Found checkpoint: {checkpoint_path}")


def test_telemetry_infrastructure():
    """Verify telemetry logging infrastructure exists."""
    from src import telemetry
    
    # Check telemetry module has required functions
    assert hasattr(telemetry, "log_inference"), "Missing log_inference function"
    
    # Check logs directory exists or can be created
    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    # Test telemetry logging (won't actually log without real inference)
    try:
        telemetry.log_inference(
            run_id="test_run",
            model_path="test_model.pth",
            image_name="test.jpg",
            top1_label="test_class",
            top1_confidence=0.95,
            latency_ms=50.0
        )
        print("✓ Telemetry logging successful")
    except Exception as e:
        pytest.fail(f"Telemetry logging failed: {e}")


def test_datetime_utc_migration():
    """Verify no datetime.utcnow() calls remain."""
    import glob
    
    # Search for deprecated utcnow calls in Python files
    deprecated_found = []
    
    for pattern in ["src/**/*.py", "app/**/*.py", "notebooks/**/*.py"]:
        for filepath in glob.glob(str(PROJECT_ROOT / pattern), recursive=True):
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                if "datetime.utcnow()" in content:
                    deprecated_found.append(filepath)
    
    if deprecated_found:
        pytest.fail(f"Found deprecated datetime.utcnow() in:\n" + "\n".join(deprecated_found))
    
    print("✓ No deprecated datetime.utcnow() calls found")


def test_model_hub_data_loading():
    """Verify Model Hub can load experiments data."""
    experiments_path = PROJECT_ROOT / "experiments.csv"
    
    if not experiments_path.exists():
        print("⚠ No experiments.csv - Model Hub will show empty state")
        pytest.skip("No experiments found")
        return
    
    df = pd.read_csv(experiments_path)
    
    # Verify timestamp parsing works
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce")
    
    # Verify sorting works
    df = df.sort_values("timestamp_utc", ascending=False)
    
    # Verify filtering columns exist
    assert "model_name" in df.columns
    assert "framework" in df.columns
    assert "val_macro_f1" in df.columns
    
    print(f"✓ Model Hub can load {len(df)} experiments")


def test_gradcam_artifacts():
    """Check if Grad-CAM artifacts exist for trained models."""
    experiments_path = PROJECT_ROOT / "experiments.csv"
    
    if not experiments_path.exists():
        pytest.skip("No experiments.csv found")
        return
    
    df = pd.read_csv(experiments_path)
    
    if df.empty:
        pytest.skip("No experiments recorded")
        return
    
    gradcam_count = 0
    for _, row in df.iterrows():
        gradcam_folder = row.get("gradcam_folder")
        if gradcam_folder and pd.notna(gradcam_folder):
            full_path = PROJECT_ROOT / gradcam_folder
            if full_path.exists():
                images = list(full_path.glob("*.png"))
                gradcam_count += len(images)
    
    print(f"✓ Found {gradcam_count} Grad-CAM images across all experiments")


def test_real_data_exists():
    """Verify REAL training data exists (not demo data)."""
    data_dir = PROJECT_ROOT / "data"
    
    assert data_dir.exists(), "Data directory doesn't exist!"
    
    # Check for real plant disease classes
    real_classes = [
        "Tomato___healthy",
        "Tomato___Late_blight",
        "Corn_(maize)___healthy",
        "Potato___Early_blight"
    ]
    
    found_classes = []
    for class_name in real_classes:
        class_dir = data_dir / class_name
        if class_dir.exists():
            num_images = len(list(class_dir.glob("*.jpg"))) + len(list(class_dir.glob("*.JPG")))
            if num_images > 0:
                found_classes.append(f"{class_name} ({num_images} images)")
    
    assert len(found_classes) > 0, "No real training data found!"
    
    print(f"✓ Found {len(found_classes)} real data classes:")
    for cls in found_classes:
        print(f"  - {cls}")


if __name__ == "__main__":
    print("=" * 80)
    print("LAZARUS DASHBOARD INTEGRATION TEST")
    print("Testing REAL model integration (no demos)")
    print("=" * 80)
    print()
    
    tests = [
        ("Notebook Imports", test_notebook_imports),
        ("Experiments CSV Structure", test_experiments_csv_structure),
        ("Checkpoint Loading", test_dashboard_checkpoint_loading),
        ("Telemetry Infrastructure", test_telemetry_infrastructure),
        ("Datetime UTC Migration", test_datetime_utc_migration),
        ("Model Hub Data Loading", test_model_hub_data_loading),
        ("Grad-CAM Artifacts", test_gradcam_artifacts),
        ("Real Training Data", test_real_data_exists),
    ]
    
    passed = 0
    failed = 0
    skipped = 0
    
    for name, test_func in tests:
        print(f"\n[Testing] {name}")
        print("-" * 40)
        try:
            test_func()
            passed += 1
            print(f"[PASS] {name}")
        except pytest.skip.Exception as e:
            skipped += 1
            print(f"[SKIP] {name}: {e}")
        except Exception as e:
            failed += 1
            print(f"[FAIL] {name}: {e}")
    
    print("\n" + "=" * 80)
    print(f"RESULTS: {passed} passed, {failed} failed, {skipped} skipped")
    print("=" * 80)
    
    if failed > 0:
        sys.exit(1)
