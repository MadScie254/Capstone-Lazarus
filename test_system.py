#!/usr/bin/env python3
"""
Test script to validate all fixes are working
"""
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append('src')

def test_imports():
    """Test if all modules can be imported"""
    print("🧪 Testing imports...")
    
    try:
        from data_utils import PlantDiseaseDataLoader
        print("✅ data_utils imported successfully")
    except Exception as e:
        print(f"❌ data_utils import failed: {e}")
        return False
    
    try:
        from model_factory import ModelFactory
        print("✅ model_factory imported successfully")
    except Exception as e:
        print(f"❌ model_factory import failed: {e}")
        return False
    
    try:
        # Import directly from inference.py file
        import inference as inf
        PlantDiseaseInference = inf.PlantDiseaseInference
        print("✅ inference imported successfully")
    except Exception as e:
        print(f"❌ inference import failed: {e}")
        return False
    
    return True

def test_data_loading():
    """Test data loading functionality"""
    print("\n📊 Testing data loading...")
    
    try:
        from data_utils import PlantDiseaseDataLoader
        
        # Test data loader creation
        loader = PlantDiseaseDataLoader('data', img_size=(224, 224))
        print("✅ DataLoader created")
        
        # Test dataset scanning
        stats = loader.scan_dataset()
        print(f"✅ Dataset scanned: {stats['num_classes']} classes, {stats['total_images']:,} images")
        
        return True
        
    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        return False

def test_model_creation():
    """Test model creation"""
    print("\n🤖 Testing model creation...")
    
    try:
        from model_factory import ModelFactory
        
        # Create model factory
        factory = ModelFactory(input_shape=(224, 224, 3), num_classes=19)
        print("✅ ModelFactory created")
        
        # Test model creation
        model = factory.get_model('efficientnet_v2_b0', dropout_rate=0.3)
        print(f"✅ Model created: {model.name}")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
        
        # Test available architectures
        archs = factory.get_available_architectures()
        print(f"✅ Available architectures: {len(archs)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_notebook_readiness():
    """Test if notebooks can be run"""
    print("\n📓 Testing notebook readiness...")
    
    notebooks_dir = Path('notebooks')
    if not notebooks_dir.exists():
        print("❌ Notebooks directory not found")
        return False
    
    # Check if notebooks exist
    required_notebooks = [
        'eda_plant_diseases.ipynb',
        'model_training.ipynb'
    ]
    
    for notebook in required_notebooks:
        if (notebooks_dir / notebook).exists():
            print(f"✅ {notebook} found")
        else:
            print(f"❌ {notebook} not found")
            return False
    
    return True

def main():
    """Run all tests"""
    print("🌱 CAPSTONE-LAZARUS: System Validation Test")
    print("=" * 50)
    
    tests = [
        ("Import Tests", test_imports),
        ("Data Loading Tests", test_data_loading),
        ("Model Creation Tests", test_model_creation),
        ("Notebook Readiness", test_notebook_readiness)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"\n❌ {test_name} FAILED")
        except Exception as e:
            print(f"\n❌ {test_name} FAILED with error: {e}")
    
    print("\n" + "=" * 50)
    print(f"🎯 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! System is ready!")
        print("\n📋 Next Steps:")
        print("1. 📊 Run EDA notebook: jupyter notebook notebooks/eda_plant_diseases.ipynb")
        print("2. 🎯 Run training: jupyter notebook notebooks/model_training.ipynb")
        print("3. 🚀 Launch dashboard: streamlit run app/streamlit_app/main.py")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)