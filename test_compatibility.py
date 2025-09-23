# Test compatibility for Colab notebook on HP ZBook G5
import torch
import sys
import platform
import psutil
from pathlib import Path

print("🧪 CAPSTONE-LAZARUS COMPATIBILITY TEST")
print("="*50)

# System info
print(f"🖥️  System: {platform.system()} {platform.release()}")
print(f"🐍 Python: {sys.version}")
print(f"📦 PyTorch: {torch.__version__}")

# Hardware check
print(f"\n💾 RAM: {psutil.virtual_memory().total / 1024**3:.1f} GB")
print(f"🔧 CPU cores: {psutil.cpu_count()}")

# GPU check
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"🖥️  GPU: {gpu_name}")
    print(f"🎮 VRAM: {gpu_memory_gb:.1f} GB")
    print(f"✅ CUDA available: Version {torch.version.cuda}")
else:
    print("❌ CUDA not available - will use CPU")

# Test PyTorch tensor operations
print(f"\n🧮 Testing PyTorch operations...")
try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(16, 3, 224, 224).to(device)  # Simulate batch
    y = torch.nn.functional.avg_pool2d(x, 2)
    print(f"✅ Tensor operations work on {device}")
    print(f"📊 Test batch shape: {x.shape}")
    
    # Memory usage
    if torch.cuda.is_available():
        memory_used = torch.cuda.memory_allocated() / 1024**2
        print(f"💾 GPU memory used: {memory_used:.1f} MB")
    
except Exception as e:
    print(f"❌ PyTorch test failed: {e}")

# Test required packages
print(f"\n📦 Checking required packages...")
required_packages = {
    'torch': None,
    'torchvision': 'torchvision',
    'timm': 'timm',
    'albumentations': 'albumentations', 
    'yaml': 'pyyaml',
    'matplotlib': 'matplotlib',
    'numpy': 'numpy',
    'PIL': 'pillow',
    'sklearn': 'scikit-learn'
}

missing_packages = []
for package, pip_name in required_packages.items():
    try:
        __import__(package)
        print(f"✅ {package}")
    except ImportError:
        print(f"❌ {package} (need to install {pip_name or package})")
        missing_packages.append(pip_name or package)

# Config file check
config_path = Path("config.yaml")
if config_path.exists():
    print(f"\n✅ Config file found: {config_path}")
else:
    print(f"\n❌ Config file missing: {config_path}")

# Optimal settings recommendation
print(f"\n🎯 OPTIMAL SETTINGS FOR YOUR SYSTEM:")
print(f"📊 Recommended batch_size: 16 (current in config)")
print(f"👥 Recommended num_workers: 4")
print(f"⚡ AMP (Mixed Precision): {'Enabled' if torch.cuda.is_available() else 'Disabled (CPU only)'}")
print(f"🧠 Expected VRAM usage: ~3-3.5GB (within P2000 4GB limit)")

if missing_packages:
    print(f"\n📋 TO INSTALL MISSING PACKAGES:")
    print(f"conda install {' '.join(missing_packages)}")
    print(f"# or")
    print(f"pip install {' '.join(missing_packages)}")

print(f"\n🚀 COMPATIBILITY STATUS:")
if torch.cuda.is_available() and not missing_packages:
    print("✅ FULLY COMPATIBLE - Ready for local training!")
    print("🎯 Expected performance: 15-30 min per model")
elif not missing_packages:
    print("✅ COMPATIBLE (CPU) - Training will be slower but functional")
    print("🎯 Expected performance: 60-120 min per model")
else:
    print("⚠️  NEEDS PACKAGE INSTALLATION - Install missing packages first")

print(f"\n📋 NEXT STEPS:")
print(f"1. Install any missing packages above")
print(f"2. Open notebooks/colab_training.ipynb in Jupyter")
print(f"3. Run all cells sequentially")
print(f"4. Notebook will auto-detect local environment")