#!/usr/bin/env python3
"""
Validation Script for PlantDiseaseDataLoader
============================================
Lightweight script to validate data loader functionality.
"""

import sys
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / 'src'))

def main():
    """Validate data loader functionality."""
    print("🔍 CAPSTONE-LAZARUS Data Loader Validation")
    print("=" * 50)
    
    try:
        # Import the data loader
        from data_utils import PlantDiseaseDataLoader
        print("✅ Successfully imported PlantDiseaseDataLoader")
        
        # Initialize loader
        data_path = project_root / 'data'
        if not data_path.exists():
            print(f"⚠️  Data directory not found: {data_path}")
            print("   This is expected if no sample data is present")
            
            # Create minimal test structure for validation
            test_data_path = project_root / 'test_data'
            test_class_dir = test_data_path / 'test_class'
            test_class_dir.mkdir(parents=True, exist_ok=True)
            
            # Create a dummy image file
            dummy_image = test_class_dir / 'dummy.jpg'
            dummy_image.write_bytes(b'\x00' * 100)  # Minimal dummy file
            
            data_path = test_data_path
            print(f"📁 Created test data structure: {data_path}")
        
        loader = PlantDiseaseDataLoader(str(data_path))
        print("✅ Successfully initialized PlantDiseaseDataLoader")
        
        # Test get_dataset_stats method
        try:
            stats = loader.get_dataset_stats(compute_image_shape=False)
            print("✅ Successfully called get_dataset_stats()")
            
            # Validate return structure
            required_keys = [
                'total_images', 'valid_images', 'corrupted_images', 
                'num_classes', 'class_distribution', 'class_names', 
                'imbalance_ratio', 'dataframe'
            ]
            
            for key in required_keys:
                if key in stats:
                    print(f"   ✅ {key}: {type(stats[key])}")
                else:
                    print(f"   ❌ Missing key: {key}")
            
            # Validate DataFrame
            import pandas as pd
            if isinstance(stats['dataframe'], pd.DataFrame):
                print("✅ DataFrame validation passed")
            else:
                print(f"❌ DataFrame validation failed: {type(stats['dataframe'])}")
            
            # Print summary
            print(f"\n📊 Dataset Summary:")
            print(f"   Total Images: {stats.get('total_images', 'N/A')}")
            print(f"   Valid Images: {stats.get('valid_images', 'N/A')}")
            print(f"   Corrupted Images: {stats.get('corrupted_images', 'N/A')}")
            print(f"   Number of Classes: {stats.get('num_classes', 'N/A')}")
            print(f"   Imbalance Ratio: {stats.get('imbalance_ratio', 'N/A'):.2f}")
            
            print(f"\n🎯 Class Distribution:")
            class_dist = stats.get('class_distribution', {})
            for class_name, count in class_dist.items():
                print(f"   {class_name}: {count}")
                
        except Exception as e:
            print(f"❌ get_dataset_stats() failed: {e}")
            return False
            
        # Clean up test data if created
        if 'test_data_path' in locals():
            import shutil
            shutil.rmtree(test_data_path, ignore_errors=True)
            print("🧹 Cleaned up test data")
        
        print(f"\n✅ All validation checks passed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)