# 🌱 CAPSTONE-LAZARUS: AI Plant Disease Detector

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.15+](https://img.shields.io/badge/tensorflow-2.15+-orange.svg)](https://tensorflow.org/)
[![CI Pipeline](https://img.shields.io/badge/CI-passing-brightgreen.svg)](https://github.com/your-repo/actions)
[![Code Quality](https://img.shields.io/badge/code%20quality-validated-brightgreen.svg)](#validation)

**CAPSTONE-LAZARUS** is a robust, production-ready computer vision system that detects plant diseases from leaf images using deep learning. Designed for **farmers, agronomists, and agricultural support systems** to enable **quick diagnosis** and **corrective action**.

## 🚀 Recent Improvements (2025)

✅ **Python 3.11+ Compatibility** with pinned dependencies  
✅ **Defensive Programming** throughout data pipeline  
✅ **Deterministic Training** with reproducible seeds  
✅ **Comprehensive Testing** with CI/CD pipeline  
✅ **Clean EDA Notebook** replacing corrupted version  
✅ **Robust Model Loading** with proper error handling  
✅ **Environment Validation** scripts for quick setup verification  

### 🔧 Quick Validation
```bash
# Validate environment setup
python scripts/validate_environment.py

# Test data loader functionality  
python scripts/validate_loader.py

# Test inference functions
python scripts/validate_inference.py

# Run smoke tests
python tests/test_data_utils.py
```

---

## 🎯 Objectives

- **Robust multi-class classification** across 19 plant disease classes
- **High recall on critical diseases** (>90%) to minimize false negatives
- **Interpretable predictions** with confidence scores and Grad-CAM visualizations
- **Lightweight deployment** for mobile/edge devices
- **Interactive dashboard** with Plotly visualizations

---

## 📊 Dataset

**26,134 RGB leaf images** across **19 disease classes**:

### 🌽 **Corn (Maize)**

- Healthy
- Cercospora leaf spot / Gray leaf spot
- Common rust  
- Northern Leaf Blight (+ oversampled/undersampled variants)

### 🥔 **Potato**

- Healthy
- Early blight
- Late blight

### 🍅 **Tomato**

- Healthy
- Bacterial spot
- Early blight
- Late blight
- Leaf Mold
- Septoria leaf spot
- Spider mites Two-spotted spider mite
- Target Spot
- Tomato mosaic virus
- Tomato Yellow Leaf Curl Virus

---

## 🛠️ Architecture

```txt
📂 Capstone-Lazarus/
├── 📊 data/                     # 26K+ plant disease images
├── 📓 notebooks/                # Jupyter notebooks for EDA & training
│   ├── eda_plant_diseases.ipynb      # Comprehensive data exploration  
│   └── model_training.ipynb          # Model training & evaluation
├── 🧠 models/                   # Saved trained models
├── 📱 app/                      # Streamlit dashboard
│   └── streamlit_app.py
├── 🔧 src/                      # Core ML modules
│   ├── data_utils.py                 # Data loading & preprocessing
│   ├── model_factory.py              # Model architectures
│   ├── training.py                   # Training pipeline
│   └── inference.py                  # Prediction & explanation
└── 📋 requirements.txt          # Dependencies
```

---

## 🚀 Quick Start

### 🎯 Balanced Subset Training (New!)

**Perfect for laptops and quick experiments!** Create a small, balanced subset of your data for rapid prototyping:

```bash
# 1. Create a balanced subset (50 samples per class)
python scripts/create_subset.py \
    --data-dir data \
    --subset-dir my_subset \
    --samples-per-class 50 \
    --seed 42

# 2. Train on subset using CLI
python src/train.py \
    --data-dir data \
    --subset-dir my_subset \
    --samples-per-class 50 \
    --epochs 10 \
    --batch-size 16

# 3. OR use the Jupyter notebook
jupyter notebook notebooks/jupyter_subset_training.ipynb

# 4. Quick test everything works
bash scripts/quick_test.sh
```

**Key Benefits:**
- ⚡ **Fast experiments** (minutes instead of hours)
- 💻 **Laptop-friendly** (works with limited GPU/CPU)
- 🎯 **Balanced sampling** (equal samples per class)
- 🔄 **Deterministic** (reproducible results)
- 🔗 **Symlink optimization** (saves disk space)

### 🎓 Full Training Workflow

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Explore data
jupyter notebook notebooks/eda_plant_diseases.ipynb

# 3. Train models  
jupyter notebook notebooks/model_training.ipynb

# 4. Launch dashboard
streamlit run app/streamlit_app.py
```

---

## � Project Structure

```
├── 📊 data/                          # Plant disease images (organized by class)
├── 🔬 notebooks/                     # Jupyter analysis & training notebooks
│   ├── eda_plant_diseases_clean.ipynb    # Data exploration (validated)
│   ├── jupyter_subset_training.ipynb     # Quick subset training (NEW!)
│   └── model_training.ipynb              # Full model training
├── 🤖 src/                          # Core source code
│   ├── data_utils.py                     # Data loading & preprocessing
│   ├── train.py                          # CLI trainer for subsets (NEW!)
│   ├── inference.py                      # Model inference utilities
│   └── model_factory.py                  # Model architectures
├── 🧪 scripts/                      # Utility scripts
│   ├── create_subset.py                  # Balanced subset creation (NEW!)
│   ├── quick_test.sh                     # End-to-end testing (NEW!)
│   └── validate_*.py                     # Environment validation
├── 🔧 tests/                        # Unit & integration tests
│   ├── test_create_subset.py             # Subset creation tests (NEW!)
│   └── test_data_utils.py                # Data utilities tests
├── 📱 app/streamlit_app/            # Web dashboard
└── 📋 requirements.txt               # Dependencies (updated with pytest)
```

---

## 🧪 Testing & Validation

**Comprehensive testing suite** to ensure reliability:

```bash
# Quick end-to-end test (recommended first step)
bash scripts/quick_test.sh

# Run unit tests
python -m pytest tests/ -v

# Test specific functionality
python -m pytest tests/test_create_subset.py -v

# Environment validation
python scripts/validate_environment.py
python scripts/validate_loader.py
python scripts/validate_inference.py
```

**Testing Coverage:**
- ✅ **Subset creation** with balanced sampling
- ✅ **File operations** (symlinks, copying, directory structure)
- ✅ **Data loading** with PyTorch and TensorFlow
- ✅ **Model training** pipeline
- ✅ **Deterministic behavior** (reproducible results)

---

## �📈 Success Metrics

- **Macro F1 Score**: ≥ 0.85
- **Critical Disease Recall**: ≥ 90%
- **Model Size**: ≤ 50MB
- **Inference Time**: ≤ 500ms

---

## 🔍 Key Features

- **Transfer Learning** with EfficientNet, ResNet, MobileNet
- **Advanced Data Augmentation** for field conditions
- **Class-weighted Loss** for imbalanced datasets
- **Grad-CAM Explanations** for model interpretability
- **Uncertainty Estimation** with Monte Carlo Dropout
- **Interactive Dashboard** with Plotly visualizations

---

## 🌟 Impact

- **Early disease detection** → reduced crop losses
- **Precision agriculture** → optimized treatment
- **Farmer empowerment** → AI-assisted decisions
- **Sustainable farming** → reduced pesticide usage