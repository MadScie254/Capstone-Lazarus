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

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Validate environment
python scripts/validate_environment.py

# 3. Explore data
jupyter notebook notebooks/eda_plant_diseases.ipynb

# 4. Train models (see Local Training & Dashboard below)
jupyter notebook notebooks/master_model_trainer.ipynb

# 5. Launch dashboard
streamlit run app/lazarus_console/__init__.py
```

---

## 🎓 Local Training & Dashboard

### Training Models

**Option 1: Jupyter Notebook (Recommended)**

```bash
# Full training run
jupyter notebook notebooks/master_model_trainer.ipynb

# Or convert and execute directly
jupyter nbconvert --execute notebooks/master_model_trainer.ipynb
```

**Option 2: Python CLI**

```bash
# Fast smoke test (1 epoch, small sample)
python -m src.master_trainer --fast-test

# Full training from config
python -m src.master_trainer --config config.yaml
```

**For Low-Spec Machines (HP ZBook 15 G5, 16GB RAM):**

Recommended settings in notebook or `config.yaml`:
- `batch_size: 2-4` (instead of 8)
- `image_size: 160` (instead of 224)  
- `fast_test: true` for initial runs
- Train one model at a time

### Launching the Dashboard

```bash
# Primary method
streamlit run app/lazarus_console/__init__.py

# Or use convenience wrapper
python run.py
```

**Dashboard Features:**
- 🏠 **Mission Readiness**: Latest metrics, checkpoints feed
- 🔬 **Inference Lab**: Batch upload, ensemble mode, confidence thresholds
- 🧬 **Explainability Studio**: Grad-CAM with opacity control
- 📊 **Model Comparison**: Side-by-side metrics, confusion matrices

### Running Tests

```bash
# Smoke test suite (PowerShell on Windows)
.\run_smoke_tests.ps1

# Or Bash (Linux/macOS/Git Bash)
./run_smoke_tests.sh

# Individual test files
pytest tests/test_smoke_train.py -v
pytest tests/test_streamlit_integration.py -v
pytest tests/test_master_trainer.py -v
```

### Profiling Inference

```bash
# CPU profiling
python scripts/profile_inference.py \
  --model models/{run_id}/best.pth \
  --device cpu \
  --runs 100

# GPU profiling (if CUDA available)
python scripts/profile_inference.py \
  --model models/{run_id}/best.pth \
  --device cuda \
  --batch-size 8
```

**For detailed troubleshooting, see [`docs/runbook.md`](docs/runbook.md).**

---

## 📈 Success Metrics

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