#!/usr/bin/env python3
"""
Environment Validation Script
============================
Validates Python environment setup and dependencies.
"""

import sys
from pathlib import Path

def check_python_version():
    """Check Python version compatibility."""
    print("🐍 Python Environment Check")
    print(f"   Version: {sys.version}")
    print(f"   Executable: {sys.executable}")
    
    if sys.version_info >= (3, 11):
        print("✅ Python 3.11+ requirement satisfied")
        return True
    else:
        print("❌ Python 3.11+ required")
        return False

def check_core_dependencies():
    """Check availability of core dependencies."""
    print("\n📦 Core Dependencies Check")
    dependencies = {
        'tensorflow': 'TensorFlow',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'matplotlib': 'Matplotlib',
        'plotly': 'Plotly',
        'pillow': 'Pillow (PIL)',
        'scikit-learn': 'Scikit-learn',
    }
    
    all_available = True
    for module, name in dependencies.items():
        try:
            if module == 'pillow':
                import PIL
                version = getattr(PIL, '__version__', 'unknown')
            elif module == 'scikit-learn':
                import sklearn
                version = getattr(sklearn, '__version__', 'unknown')
            else:
                imported = __import__(module)
                version = getattr(imported, '__version__', 'unknown')
            
            print(f"   ✅ {name}: {version}")
        except ImportError:
            print(f"   ❌ {name}: Not available")
            all_available = False
    
    return all_available

def check_dev_dependencies():
    """Check development dependencies."""
    print("\n🛠️  Development Dependencies Check")
    dev_dependencies = {
        'pytest': 'pytest',
        'black': 'Black',
        'ruff': 'Ruff',
        'notebook': 'Jupyter Notebook',
    }
    
    dev_available = True
    for module, name in dev_dependencies.items():
        try:
            imported = __import__(module)
            version = getattr(imported, '__version__', 'unknown')
            print(f"   ✅ {name}: {version}")
        except ImportError:
            print(f"   ⚠️  {name}: Not available (development dependency)")
            # Don't fail for dev dependencies
    
    return True  # Always return True for dev dependencies

def check_gpu_availability():
    """Check GPU availability for TensorFlow."""
    print("\n🎮 GPU Availability Check")
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"   ✅ {len(gpus)} GPU(s) available:")
            for i, gpu in enumerate(gpus):
                print(f"      GPU {i}: {gpu.name}")
        else:
            print("   ⚠️  No GPUs available (CPU-only mode)")
        return True
    except Exception as e:
        print(f"   ❌ GPU check failed: {e}")
        return False

def check_project_structure():
    """Check basic project structure."""
    print("\n📁 Project Structure Check")
    project_root = Path(__file__).parent.parent
    
    required_paths = [
        'src',
        'src/data_utils.py',
        'src/inference.py',
        'notebooks',
        'requirements.txt',
        'environment.yml',
    ]
    
    all_present = True
    for path_str in required_paths:
        path = project_root / path_str
        if path.exists():
            if path.is_file():
                size = path.stat().st_size
                print(f"   ✅ {path_str}: {size} bytes")
            else:
                print(f"   ✅ {path_str}/: directory")
        else:
            print(f"   ❌ {path_str}: missing")
            all_present = False
    
    return all_present

def main():
    """Run all validation checks."""
    print("🔍 CAPSTONE-LAZARUS Environment Validation")
    print("=" * 50)
    
    checks = [
        ("Python Version", check_python_version),
        ("Core Dependencies", check_core_dependencies),
        ("Development Dependencies", check_dev_dependencies),
        ("GPU Availability", check_gpu_availability),
        ("Project Structure", check_project_structure),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"   ❌ {name} check failed: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 50)
    print("📊 VALIDATION SUMMARY")
    print("=" * 50)
    
    all_critical_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
        
        # Only GPU and Dev Dependencies are non-critical
        if not passed and name not in ["GPU Availability", "Development Dependencies"]:
            all_critical_passed = False
    
    if all_critical_passed:
        print(f"\n🎉 Environment validation completed successfully!")
        print("   The repository is ready for development.")
    else:
        print(f"\n⚠️  Some critical checks failed.")
        print("   Please install missing dependencies before proceeding.")
    
    return all_critical_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)