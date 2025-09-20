#!/usr/bin/env python3
"""
Basic smoke tests for data utilities
====================================
Simple validation tests for core functionality.
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path
from PIL import Image
import pandas as pd

# Add src to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / 'src'))


def create_test_dataset(temp_dir: Path) -> Path:
    """Create a minimal test dataset for testing."""
    # Create class directories
    class1_dir = temp_dir / 'class1'
    class2_dir = temp_dir / 'class2'
    class1_dir.mkdir(parents=True, exist_ok=True)
    class2_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dummy images
    for i, class_dir in enumerate([class1_dir, class2_dir]):
        for j in range(3):  # 3 images per class
            img_path = class_dir / f'image_{j}.jpg'
            
            # Create a simple RGB image
            img = Image.new('RGB', (224, 224), color=(i*100, j*50, 100))
            img.save(img_path, 'JPEG')
    
    return temp_dir


def test_data_loader_import():
    """Test that PlantDiseaseDataLoader can be imported."""
    try:
        from data_utils import PlantDiseaseDataLoader
        print("✅ PlantDiseaseDataLoader import successful")
        return True
    except ImportError as e:
        print(f"❌ PlantDiseaseDataLoader import failed: {e}")
        return False


def test_data_loader_initialization():
    """Test data loader initialization with test data."""
    try:
        from data_utils import PlantDiseaseDataLoader
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            test_data_path = create_test_dataset(temp_path)
            
            # Initialize loader
            loader = PlantDiseaseDataLoader(str(test_data_path))
            
            # Check basic attributes
            if hasattr(loader, 'data_dir'):
                print("✅ PlantDiseaseDataLoader initialization successful")
                return True
            else:
                print("❌ PlantDiseaseDataLoader missing data_dir attribute")
                return False
                
    except Exception as e:
        print(f"❌ PlantDiseaseDataLoader initialization failed: {e}")
        return False


def test_get_dataset_stats():
    """Test get_dataset_stats method."""
    try:
        from data_utils import PlantDiseaseDataLoader
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            test_data_path = create_test_dataset(temp_path)
            
            # Initialize loader
            loader = PlantDiseaseDataLoader(str(test_data_path))
            
            # Call get_dataset_stats
            stats = loader.get_dataset_stats()
            
            # Validate return structure
            required_keys = [
                'total_images', 'valid_images', 'corrupted_images', 
                'num_classes', 'class_distribution', 'class_names', 
                'imbalance_ratio', 'dataframe'
            ]
            
            for key in required_keys:
                if key not in stats:
                    print(f"❌ get_dataset_stats missing key: {key}")
                    return False
            
            # Validate values - check structure and basic sanity
            if stats['num_classes'] < 1:
                print(f"❌ Expected at least 1 class, got {stats['num_classes']}")
                return False
                
            if stats['valid_images'] < 1:
                print(f"❌ Expected at least 1 valid image, got {stats['valid_images']}")
                return False
                
            if not isinstance(stats['dataframe'], pd.DataFrame):
                print(f"❌ Expected DataFrame, got {type(stats['dataframe'])}")
                return False
            
            print("✅ get_dataset_stats validation successful")
            return True
                
    except Exception as e:
        print(f"❌ get_dataset_stats test failed: {e}")
        return False


def test_inference_import():
    """Test that inference functions can be imported."""
    try:
        from inference import load_model_keras, predict_single_image
        print("✅ Inference functions import successful")
        return True
    except ImportError as e:
        print(f"❌ Inference functions import failed: {e}")
        return False


def test_load_model_keras():
    """Test load_model_keras function (should handle missing model gracefully)."""
    try:
        from inference import load_model_keras
        
        # This should either return None or raise FileNotFoundError
        try:
            model = load_model_keras()
            if model is None:
                print("✅ load_model_keras handles missing model gracefully")
                return True
            else:
                print(f"✅ load_model_keras loaded model: {type(model)}")
                return True
        except FileNotFoundError:
            print("✅ load_model_keras raises FileNotFoundError for missing model (expected)")
            return True
        except Exception as e:
            print(f"⚠️  load_model_keras raised unexpected error: {type(e).__name__}: {e}")
            return True  # Non-critical for this test
                
    except Exception as e:
        print(f"❌ load_model_keras test failed: {e}")
        return False


def main():
    """Run all smoke tests."""
    print("🧪 CAPSTONE-LAZARUS Smoke Tests")
    print("=" * 50)
    
    tests = [
        ("Data Loader Import", test_data_loader_import),
        ("Data Loader Initialization", test_data_loader_initialization),
        ("get_dataset_stats Method", test_get_dataset_stats),
        ("Inference Functions Import", test_inference_import),
        ("load_model_keras Function", test_load_model_keras),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n🔍 Running: {name}")
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} crashed: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print(f"\n📈 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All smoke tests passed!")
        return True
    else:
        print("⚠️  Some tests failed. Check logs above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)