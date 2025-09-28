#!/usr/bin/env python3
"""
Quick test to verify DatasetManager and ModelManager are working
"""

import sys
from pathlib import Path

# Add the console directory to path
console_root = Path(__file__).parent / "app" / "lazarus_console"
sys.path.append(str(console_root))

from utils.dataset_manager import DatasetManager
from utils.model_manager import ModelManager

def test_managers():
    """Test both managers to ensure they load real data"""
    
    print("🧪 Testing Lazarus Console Data Managers...")
    
    # Test DatasetManager
    print("\n📊 Testing DatasetManager...")
    try:
        project_root = Path(__file__).parent
        dataset_manager = DatasetManager(project_root)
        
        print(f"✅ DatasetManager initialized")
        print(f"   - Class names loaded: {len(dataset_manager.class_names)} classes")
        print(f"   - Classes: {dataset_manager.class_names[:5]}{'...' if len(dataset_manager.class_names) > 5 else ''}")
        
        # Test class statistics
        try:
            stats = dataset_manager.get_class_statistics()
            print(f"   - Statistics loaded: {len(stats)} metrics")
            if stats:
                print(f"   - Total images: {stats.get('total_images', 'N/A')}")
                print(f"   - Number of classes: {stats.get('num_classes', 'N/A')}")
        except Exception as e:
            print(f"   - Statistics error: {e}")
            
    except Exception as e:
        print(f"❌ DatasetManager error: {e}")
    
    # Test ModelManager
    print("\n🤖 Testing ModelManager...")
    try:
        project_root = Path(__file__).parent
        model_manager = ModelManager(project_root)
        
        print(f"✅ ModelManager initialized")
        print(f"   - Model registry loaded: {len(model_manager.model_registry)} models")
        
        if model_manager.model_registry:
            models = model_manager.model_registry.get('models', model_manager.model_registry)
            for model_name, model_info in models.items():
                if isinstance(model_info, dict):  # Skip non-dict entries like "version"
                    print(f"   - Model: {model_name}")
                    print(f"     - Type: {model_info.get('architecture', 'unknown')}")
                    print(f"     - Status: {model_info.get('status', 'unknown')}")
                
    except Exception as e:
        print(f"❌ ModelManager error: {e}")
    
    print("\n🎯 Test Complete!")

if __name__ == "__main__":
    test_managers()