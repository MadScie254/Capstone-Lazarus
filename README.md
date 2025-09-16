# 🌱 CAPSTONE-LAZARUS: AI Plant Disease Detector

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/tensorflow-2.x-orange.svg)](https://tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)](https://streamlit.io/)

**CAPSTONE-LAZARUS** is a computer vision system that detects plant diseases from leaf images using deep learning.  
Designed for **farmers, agronomists, and agricultural support systems** to enable **quick diagnosis** and **corrective action**.

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

# 2. Explore data
jupyter notebook notebooks/eda_plant_diseases.ipynb

# 3. Train models  
jupyter notebook notebooks/model_training.ipynb

# 4. Launch dashboard
streamlit run app/streamlit_app.py
```

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