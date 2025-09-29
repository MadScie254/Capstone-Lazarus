#!/usr/bin/env python3
"""CAPSTONE-LAZARUS orchestration CLI."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional

from src.master_trainer import MasterTrainer

PROJECT_ROOT = Path(__file__).resolve().parent
REQUIREMENTS_PATH = PROJECT_ROOT / "requirements.txt"
DATA_DIR_CANDIDATES = [PROJECT_ROOT / "Data", PROJECT_ROOT / "data"]


def _resolve_models_argument(raw: Optional[Iterable[str]]) -> Optional[List[str]]:
    if not raw:
        return None
    cleaned = [name.strip() for name in raw if name and name.strip()]
    return cleaned or None


def _locate_data_dir() -> Path:
    for candidate in DATA_DIR_CANDIDATES:
        if candidate.exists():
            return candidate
    return DATA_DIR_CANDIDATES[0]


def setup_environment() -> None:
    """Install Python dependencies."""

    print("🔧 Setting up CAPSTONE-LAZARUS environment...")
    if not REQUIREMENTS_PATH.exists():
        print("❌ requirements.txt not found. Please run from the project root.")
        sys.exit(1)

    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS_PATH)],
            check=True,
        )
        print("✅ Dependencies installed successfully!")
    except subprocess.CalledProcessError as exc:  # pragma: no cover - CLI guard
        print(f"❌ Failed to install dependencies: {exc}")
        sys.exit(1)


def run_eda() -> None:
    """Launch the exploratory data analysis notebook."""

    print("📊 Opening EDA notebook...")
    try:
        subprocess.run(
            ["jupyter", "notebook", str(PROJECT_ROOT / "notebooks" / "eda_plant_diseases.ipynb")],
            check=True,
        )
    except subprocess.CalledProcessError:
        print("⚠️ Could not launch Jupyter notebook. Please run manually:")
        print("   jupyter notebook notebooks/eda_plant_diseases.ipynb")


def run_streamlit() -> None:
    """Launch the Streamlit dashboard."""

    print("🚀 Launching Lazarus Console...")
    try:
        subprocess.run(
            ["streamlit", "run", str(PROJECT_ROOT / "app" / "lazarus_console" / "__init__.py")],
            check=True,
        )
    except subprocess.CalledProcessError:
        print("❌ Failed to launch Streamlit app")
        print("Please ensure Streamlit is installed: pip install streamlit")


def validate_data() -> bool:
    """Validate that the dataset directory exists and contains images."""

    print("🔍 Validating dataset...")
    data_path = _locate_data_dir()

    if not data_path.exists():
        print(f"❌ Data directory not found at {data_path}")
        return False

    class_dirs = [d for d in data_path.iterdir() if d.is_dir()]
    if not class_dirs:
        print(f"❌ No class folders discovered in {data_path}")
        return False

    total_images = 0
    for class_dir in class_dirs:
        image_count = sum(1 for _ in class_dir.glob("*.jp*g"))
        total_images += image_count
        print(f"   📁 {class_dir.name}: {image_count} images")

    print(f"📊 Total images: {total_images}")

    if total_images == 0:
        print("❌ No images found in dataset")
        return False

    print("✅ Dataset validation passed!")
    return True


def run_training(fast: bool = True, models: Optional[Iterable[str]] = None) -> bool:
    """Run the orchestrated training pipeline."""

    mode = "FAST-SMOKE" if fast else "FULL"
    print(f"🎯 Launching Lazarus master trainer ({mode})...")
    try:
        trainer = MasterTrainer()
        trainer.run(model_names=_resolve_models_argument(models), fast_test=fast)
        print("✅ Training pipeline completed")
        return True
    except Exception as exc:  # pragma: no cover - CLI guard
        print(f"❌ Training failed: {exc}")
        return False


def run_tests(models: Optional[Iterable[str]] = None) -> bool:
    """Run basic smoke tests across managers and the trainer."""

    print("🧪 Running Lazarus health checks...")

    try:
        from app.lazarus_console.utils.dataset_manager import DatasetManager
        from app.lazarus_console.utils.model_manager import ModelManager
    except ImportError as exc:
        print(f"❌ Console utilities unavailable: {exc}")
        return False

    project_root = PROJECT_ROOT

    try:
        dataset_manager = DatasetManager(project_root)
        stats = dataset_manager.get_class_statistics()
        print(
            "✅ DatasetManager initialized",
            f"({stats['num_classes']} classes, {stats['total_images']} images)",
        )
    except Exception as exc:
        print(f"❌ DatasetManager error: {exc}")
        return False

    try:
        model_manager = ModelManager(project_root)
        available = list(model_manager.iter_models())
        print(f"✅ ModelManager initialized ({len(available)} ready models)")
    except Exception as exc:
        print(f"❌ ModelManager error: {exc}")
        return False

    try:
        trainer = MasterTrainer()
        subset = _resolve_models_argument(models) or [trainer.models[0].name]
        trainer.run(model_names=subset, fast_test=True)
        print(f"✅ MasterTrainer smoke test completed for: {', '.join(subset)}")
    except Exception as exc:
        print(f"❌ MasterTrainer smoke test failed: {exc}")
        return False

    print("🎯 Test suite finished")
    return True


def main() -> None:
    """Main execution function."""

    parser = argparse.ArgumentParser(description="CAPSTONE-LAZARUS Setup & Run Script")
    parser.add_argument(
        "action",
        choices=["setup", "eda", "train", "dashboard", "validate", "test", "all"],
        help="Action to perform",
    )
    parser.add_argument("--models", nargs="*", help="Optional subset of models to run for train/test actions")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full training instead of the default fast smoke test",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("🌱 CAPSTONE-LAZARUS: Plant Disease Detection System")
    print("=" * 60)

    if args.action in {"setup", "all"}:
        setup_environment()

    if args.action in {"validate", "all"}:
        if not validate_data():
            print("⚠️ Dataset validation failed. Please check your data directory.")
            if args.action == "all":
                return

    if args.action in {"test", "all"}:
        if not run_tests(args.models):
            print("⚠️ Tests failed. Please check your setup.")
            if args.action == "all":
                return

    if args.action == "eda":
        run_eda()

    elif args.action == "train":
        success = run_training(fast=not args.full, models=args.models)
        if not success:
            sys.exit(1)

    elif args.action == "dashboard":
        run_streamlit()

    elif args.action == "all":
        print("\n🎉 Setup complete! Here's what you can do next:")
        print("\n📊 Explore your data:")
        print("   python run.py eda")
        print("\n🎯 Train models (fast smoke test by default):")
        print("   python run.py train")
        print("\n🧪 Deep-dive training run:")
        print("   python run.py train --full")

    print("\n🚀 Launch Lazarus Console:")
    print("   python run.py dashboard")
    print("\n✨ System is ready for immersive plant disease detection!")


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()