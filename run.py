#!/usr/bin/env python3
"""
CAPSTONE-LAZARUS: Setup & Run Script
===================================
Complete setup and execution script for the plant disease detection system.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def setup_environment():
    """Set up the Python environment and install dependencies."""
    print("🔧 Setting up CAPSTONE-LAZARUS environment...")
    
    # Check if we're in the right directory
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found. Please run from project root.")
        sys.exit(1)
    
    # Install dependencies
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True)
        print("✅ Dependencies installed successfully!")
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        sys.exit(1)

def run_eda():
    """Run the exploratory data analysis notebook."""
    print("📊 Running EDA notebook...")
    try:
        subprocess.run([
            "jupyter", "notebook", "notebooks/eda_plant_diseases.ipynb"
        ], check=True)
    except subprocess.CalledProcessError:
        print("⚠️ Could not launch Jupyter notebook. Please run manually:")
        print("   jupyter notebook notebooks/eda_plant_diseases.ipynb")

def run_training():
    """Run the model training notebook."""
    print("🎯 Running training notebook...")
    try:
        subprocess.run([
            "jupyter", "notebook", "notebooks/model_training.ipynb"
        ], check=True)
    except subprocess.CalledProcessError:
        print("⚠️ Could not launch Jupyter notebook. Please run manually:")
        print("   jupyter notebook notebooks/model_training.ipynb")

def run_streamlit():
    """Launch the Streamlit dashboard."""
    print("🚀 Launching Lazarus Console...")
    try:
        subprocess.run([
            "streamlit", "run", "app/lazarus_console/__init__.py"
        ], check=True)
    except subprocess.CalledProcessError:
        print("❌ Failed to launch Streamlit app")
        print("Please ensure Streamlit is installed: pip install streamlit")

def validate_data():
    """Validate the dataset structure."""
    print("🔍 Validating dataset...")
    
    data_path = Path("data")
    if not data_path.exists():
        print("❌ Data directory not found!")
        return False
    
    # Check for expected folders
    expected_classes = [
        "Corn_(maize)___healthy",
        "Corn_(maize)___Common_rust_",
        "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
        "Corn_(maize)___Northern_Leaf_Blight",
        "Potato___healthy",
        "Potato___Early_blight",
        "Potato___Late_blight",
        "Tomato___healthy",
        "Tomato___Bacterial_spot",
        "Tomato___Early_blight",
        "Tomato___Late_blight",
        "Tomato___Leaf_Mold",
        "Tomato___Septoria_leaf_spot",
        "Tomato___Spider_mites Two-spotted_spider_mite",
        "Tomato___Target_Spot",
        "Tomato___Tomato_mosaic_virus",
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus"
    ]
    
    found_classes = [d.name for d in data_path.iterdir() if d.is_dir()]
    total_images = 0
    
    print(f"📂 Found {len(found_classes)} class directories:")
    for class_dir in found_classes:
        class_path = data_path / class_dir
        image_count = len(list(class_path.glob("*.jpg")) + list(class_path.glob("*.JPG")))
        total_images += image_count
        print(f"   📁 {class_dir}: {image_count} images")
    
    print(f"📊 Total images: {total_images}")
    
    if total_images > 0:
        print("✅ Dataset validation passed!")
        return True
    else:
        print("❌ No images found in dataset")
        return False

def run_tests():
    """Run basic functionality tests."""
    print("🧪 Running basic tests...")
    
    # Test imports
    try:
        sys.path.append('src')
        from data_utils import PlantDiseaseDataLoader
        from model_factory import ModelFactory
        print("✅ Core modules import successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Test data loading
    try:
        data_loader = PlantDiseaseDataLoader("data")
        class_names = data_loader.get_class_names()
        print(f"✅ Data loader works: {len(class_names)} classes found")
    except Exception as e:
        print(f"❌ Data loader error: {e}")
        return False
    
    return True

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="CAPSTONE-LAZARUS Setup & Run Script")
    parser.add_argument("action", choices=[
        "setup", "eda", "train", "dashboard", "validate", "test", "all"
    ], help="Action to perform")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🌱 CAPSTONE-LAZARUS: Plant Disease Detection System")
    print("=" * 60)
    
    if args.action == "setup" or args.action == "all":
        setup_environment()
    
    if args.action == "validate" or args.action == "all":
        if not validate_data():
            print("⚠️ Dataset validation failed. Please check your data directory.")
            if args.action == "all":
                return
    
    if args.action == "test" or args.action == "all":
        if not run_tests():
            print("⚠️ Tests failed. Please check your setup.")
            if args.action == "all":
                return
    
    if args.action == "eda":
        run_eda()
    
    elif args.action == "train":
        run_training()
    
    elif args.action == "dashboard":
        run_streamlit()
    
    elif args.action == "all":
        print("\n🎉 Setup complete! Here's what you can do next:")
        print("\n📊 Explore your data:")
        print("   python run.py eda")
        print("\n🎯 Train models:")
        print("   python run.py train") 
    print("\n🚀 Launch Lazarus Console:")
    print("   python run.py dashboard")
    print("\n✨ System is ready for immersive plant disease detection!")

if __name__ == "__main__":
    main()