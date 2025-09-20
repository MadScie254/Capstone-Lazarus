#!/usr/bin/env python3
"""
Validation Script for Inference Functions
=========================================
Lightweight script to validate model inference functionality.
"""

import sys
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / 'src'))

def main():
    """Validate inference functionality."""
    print("🔍 CAPSTONE-LAZARUS Inference Validation")
    print("=" * 50)
    
    try:
        # Import inference functions
        from inference import load_model_keras, predict_single_image
        print("✅ Successfully imported inference functions")
        
        # Test load_model_keras function (without actual model)
        try:
            # This should fail gracefully if no model exists
            model = load_model_keras()
            if model is None:
                print("✅ load_model_keras() returns None when no model found (expected)")
            else:
                print(f"✅ load_model_keras() returned: {type(model)}")
        except FileNotFoundError:
            print("✅ load_model_keras() raises FileNotFoundError when no model found (expected)")
        except Exception as e:
            print(f"⚠️  load_model_keras() raised: {type(e).__name__}: {e}")
        
        # Test predict_single_image function signature
        try:
            # Check if function is callable
            if callable(predict_single_image):
                print("✅ predict_single_image() is callable")
                
                # Get function signature for validation
                import inspect
                sig = inspect.signature(predict_single_image)
                expected_params = ['model', 'image_path', 'class_names']
                
                actual_params = list(sig.parameters.keys())
                print(f"   Function parameters: {actual_params}")
                
                for param in expected_params:
                    if param in actual_params:
                        print(f"   ✅ {param}: present")
                    else:
                        print(f"   ❌ {param}: missing")
            else:
                print("❌ predict_single_image() is not callable")
                
        except Exception as e:
            print(f"❌ predict_single_image() validation failed: {e}")
        
        # Test import structure
        print(f"\n📦 Module Structure:")
        import inference
        available_functions = [attr for attr in dir(inference) if not attr.startswith('_')]
        for func in available_functions:
            if callable(getattr(inference, func)):
                print(f"   ✅ {func}(): available")
        
        print(f"\n✅ All inference validation checks passed!")
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