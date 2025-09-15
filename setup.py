"""
Setup script for Plant Disease Detection System
This script helps resolve common issues and sets up the environment
"""

import subprocess
import sys
import os
import importlib.util

def check_python_version():
    """Check if Python version is compatible"""
    python_version = sys.version_info
    print(f"🐍 Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major == 3 and python_version.minor >= 8:
        print("✅ Python version is compatible")
        return True
    else:
        print("❌ Python version must be 3.8 or higher")
        return False

def check_package_installation(package_name):
    """Check if a package is installed"""
    spec = importlib.util.find_spec(package_name)
    return spec is not None

def install_package(package_name):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"✅ {package_name} installed successfully")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package_name}")
        return False

def setup_tensorflow():
    """Set up TensorFlow with proper version"""
    print("\n🔄 Setting up TensorFlow...")
    
    # Check if TensorFlow is installed
    if check_package_installation("tensorflow"):
        try:
            import tensorflow as tf
            tf_version = tf.__version__
            print(f"📦 TensorFlow version: {tf_version}")
            
            # Check if version is problematic
            if tf_version.startswith("2.20"):
                print("⚠️ TensorFlow 2.20.0 has SavedModel compatibility issues")
                print("🔄 Attempting to downgrade to TensorFlow 2.15.0...")
                
                # Uninstall current version
                subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "tensorflow", "-y"])
                
                # Install compatible version
                if install_package("tensorflow==2.15.0"):
                    print("✅ TensorFlow downgraded successfully")
                    return True
                else:
                    print("❌ Failed to downgrade TensorFlow")
                    return False
            else:
                print("✅ TensorFlow version is compatible")
                return True
                
        except Exception as e:
            print(f"❌ Error checking TensorFlow: {e}")
            return False
    else:
        print("📦 TensorFlow not found. Installing...")
        return install_package("tensorflow==2.15.0")

def setup_required_packages():
    """Set up all required packages"""
    print("\n🔄 Checking required packages...")
    
    required_packages = [
        "streamlit",
        "pillow",
        "numpy",
        "pandas",
        "plotly",
        "matplotlib",
        "seaborn",
        "opencv-python"
    ]
    
    failed_packages = []
    
    for package in required_packages:
        if not check_package_installation(package):
            print(f"📦 Installing {package}...")
            if not install_package(package):
                failed_packages.append(package)
        else:
            print(f"✅ {package} is already installed")
    
    if failed_packages:
        print(f"\n❌ Failed to install: {', '.join(failed_packages)}")
        return False
    else:
        print("\n✅ All required packages are installed")
        return True

def test_imports():
    """Test if all packages can be imported"""
    print("\n🔄 Testing package imports...")
    
    test_packages = [
        ("streamlit", "st"),
        ("tensorflow", "tf"),
        ("PIL", "PIL"),
        ("numpy", "np"),
        ("pandas", "pd"),
        ("plotly.express", "px"),
        ("matplotlib.pyplot", "plt"),
        ("seaborn", "sns")
    ]
    
    failed_imports = []
    
    for package_name, import_name in test_packages:
        try:
            __import__(package_name)
            print(f"✅ {package_name} imported successfully")
        except ImportError as e:
            print(f"❌ Failed to import {package_name}: {e}")
            failed_imports.append(package_name)
    
    if failed_imports:
        print(f"\n❌ Import failures: {', '.join(failed_imports)}")
        return False
    else:
        print("\n✅ All packages imported successfully")
        return True

def check_model_files():
    """Check if model files exist"""
    print("\n🔄 Checking model files...")
    
    model_path = "./inception_lazarus"
    required_files = [
        "saved_model.pb",
        "keras_metadata.pb"
    ]
    
    if not os.path.exists(model_path):
        print(f"❌ Model directory not found: {model_path}")
        return False
    
    for file in required_files:
        file_path = os.path.join(model_path, file)
        if os.path.exists(file_path):
            print(f"✅ {file} found")
        else:
            print(f"❌ {file} not found")
            return False
    
    variables_path = os.path.join(model_path, "variables")
    if os.path.exists(variables_path):
        print("✅ variables/ directory found")
    else:
        print("❌ variables/ directory not found")
        return False
    
    print("✅ All model files are present")
    return True

def create_requirements_file():
    """Create a requirements.txt file with compatible versions"""
    print("\n🔄 Creating requirements.txt...")
    
    requirements_content = """# Plant Disease Detection System Requirements
streamlit>=1.28.0
tensorflow==2.15.0
pillow>=9.0.0
numpy>=1.21.0
pandas>=1.3.0
plotly>=5.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
opencv-python>=4.5.0
"""
    
    try:
        with open("requirements_fixed.txt", "w") as f:
            f.write(requirements_content)
        print("✅ requirements_fixed.txt created successfully")
        print("💡 To use this file, run: pip install -r requirements_fixed.txt")
        return True
    except Exception as e:
        print(f"❌ Failed to create requirements file: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Plant Disease Detection System - Setup Script")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        print("\n❌ Setup failed: Incompatible Python version")
        return False
    
    # Set up TensorFlow
    tf_success = setup_tensorflow()
    
    # Set up other packages
    packages_success = setup_required_packages()
    
    # Test imports
    imports_success = test_imports()
    
    # Check model files
    model_files_success = check_model_files()
    
    # Create fixed requirements file
    create_requirements_file()
    
    print("\n" + "=" * 60)
    print("🏁 SETUP SUMMARY")
    print("=" * 60)
    
    print(f"🐍 Python Version: {'✅' if check_python_version() else '❌'}")
    print(f"🧠 TensorFlow: {'✅' if tf_success else '❌'}")
    print(f"📦 Packages: {'✅' if packages_success else '❌'}")
    print(f"📥 Imports: {'✅' if imports_success else '❌'}")
    print(f"🤖 Model Files: {'✅' if model_files_success else '❌'}")
    
    if all([tf_success, packages_success, imports_success, model_files_success]):
        print("\n🎉 Setup completed successfully!")
        print("\n🚀 To start the application:")
        print("   streamlit run app_working.py")
        print("\n💡 If you encounter model loading issues:")
        print("   The app_working.py includes a mock mode for demonstration")
        return True
    else:
        print("\n⚠️ Setup completed with some issues")
        print("\n🔧 Troubleshooting:")
        if not tf_success:
            print("   - Try: pip install tensorflow==2.15.0")
        if not packages_success:
            print("   - Try: pip install -r requirements_fixed.txt")
        if not model_files_success:
            print("   - Ensure the inception_lazarus model directory is present")
        return False

if __name__ == "__main__":
    main()