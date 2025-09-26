# 🔍 Capstone-Lazarus Repository Audit

**Audit Date**: 2025-09-26  
**Target**: HP ZBook Quadro P2000 (4GB VRAM, 16GB RAM)  
**Mission**: Surgical micro-job pipeline for ruthless training efficiency  

---

## 📊 Dataset Overview

| Metric | Value | Notes |
|--------|-------|-------|
| **Total Classes** | 19 | Plant disease categories |
| **Estimated Total Images** | ~26,000+ | Based on sample counts |
| **Largest Class** | Tomato_Yellow_Leaf_Curl_Virus | ~5,357 images |
| **Smallest Class** | Potato_healthy | ~152 images |
| **Class Imbalance Ratio** | ~35:1 | Requires weighted sampling |
| **Image Format** | JPG | RGB, varying sizes |
| **Storage Pattern** | Class directories | Standard ImageFolder structure |

### 📁 Class Distribution (Sample)
```
Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot: 513
Corn_(maize)___Common_rust_: 1192
Corn_(maize)___healthy: 1162
Corn_(maize)___Northern_Leaf_Blight: 985
Corn_(maize)___Northern_Leaf_Blight_oversampled: 985
[... 14 more classes]
```

---

## 🗂️ Repository Structure Analysis

### **Critical Paths**
```
📁 Capstone-Lazarus/
├── 📁 data/                    # Raw image dataset (19 classes)
├── 📁 notebooks/              # ⚠️ CHOKE POINT ZONE
│   ├── model_training.ipynb          # PRIMARY TARGET
│   ├── colab_training.ipynb          # SECONDARY TARGET  
│   ├── eda_plant_diseases_clean.ipynb # EDA reference
│   └── comprehensive_*.ipynb         # Legacy monoliths
├── 📁 src/                     # Python utilities
│   ├── data_utils.py               # TF/Keras data loader
│   ├── data_utils_torch.py         # PyTorch data loader  
│   └── [other modules]
├── 📁 app/                     # Streamlit dashboard
├── 📁 models/                  # Empty (opportunity)
├── 📁 experiments/            # Empty (opportunity)  
└── 📁 features/               # Missing (will create)
```

### **Artifact Inventory**
| Directory | Status | Usage | Action Plan |
|-----------|--------|-------|-------------|
| `data/` | ✅ Populated | 19 plant disease classes | Use as-is |
| `models/` | ⚠️ Empty | No cached models | Create registry + checkpoints |
| `experiments/` | ⚠️ Empty | No experiment tracking | Create SQLite + metadata |
| `features/` | ❌ Missing | No feature cache | **Create micro-job extraction** |
| `notebooks/` | 🔥 Bloated | Monolithic training | **Surgical segmentation** |

---

## 🚨 Current Training Bottlenecks

### **Memory/VRAM Issues** 
- **No feature caching** → Full image preprocessing per epoch
- **Large batch loading** → OOM on 4GB VRAM  
- **Full-resolution processing** → Unnecessary memory usage
- **No gradient accumulation** → Suboptimal batch sizes

### **Training Speed Issues**
- **End-to-end pipelines** → No resumable micro-jobs
- **No head-only training** → Encoder re-computation waste
- **Heavy augmentation** → CPU bottleneck during training
- **No experiment management** → Manual hyperparameter tracking

### **Development Friction**
- **Monolithic notebooks** → Hours-long uninterruptible runs
- **No segmentation capability** → Can't handle large images
- **Manual model selection** → No systematic ablation
- **No serving infrastructure** → Heavy Streamlit model loading

---

## 🎯 Files Requiring Modification

### **HIGH PRIORITY - Immediate Targets**

1. **`notebooks/model_training.ipynb`**
   - Current: Monolithic TF/Keras training pipeline
   - Problem: 979 training batches × epochs = long blocking runs
   - Solution: Convert to feature-cache + head-only micro-jobs

2. **`notebooks/colab_training.ipynb`** 
   - Current: PyTorch training with full data loading
   - Problem: Heavy augmentation + full image loading per batch
   - Solution: Spatial feature extraction + patch training

3. **`src/data_utils.py`**
   - Current: TensorFlow ImageDataGenerator approach
   - Problem: No feature caching, heavy augmentation overhead
   - Enhancement: Add feature extraction utilities

### **MEDIUM PRIORITY - New Capabilities**

4. **Create `notebooks/02_feature_extract_microjobs.ipynb`**
   - Purpose: Chunk encoder extraction into resumable jobs
   - Target: Process 64 images per job by default

5. **Create `notebooks/03_train_head_fastloop.ipynb`**
   - Purpose: Head-only training on cached features  
   - Target: 3 ablation experiments < 10 minutes total

### **LOW PRIORITY - Infrastructure**

6. **`app/streamlit_app.py`**
   - Current: Direct model loading in UI thread
   - Enhancement: API-based inference with student models

---

## 💾 Proposed File Structure (Post-Implementation)

```
📁 Capstone-Lazarus/
├── 📁 features/                    # 🆕 Feature cache
│   ├── manifest_features.v001.csv     # Feature inventory
│   ├── jobs_queue.csv                  # Micro-job queue  
│   ├── encoder_efficientnet_b0/        # Per-encoder features
│   │   ├── img_00001.npz              # float16 cached features
│   │   ├── img_00002.npz              
│   │   └── [...]
│   └── _job_*.done                     # Job completion flags
├── 📁 experiments/                # 🆕 Experiment tracking
│   ├── experiments.db                 # SQLite experiment log
│   ├── exp_20250926_143022/           # Timestamped experiments
│   │   ├── config.yaml
│   │   ├── checkpoints/best_head.pt
│   │   ├── metrics.json
│   │   └── model_card.json
│   └── [...]
├── 📁 models/                     # Enhanced model registry
│   ├── registry.json                  # Model catalog
│   ├── teacher/teacher_v1.pt          # Full models
│   └── student/student_v1.onnx        # Exported students
├── 📁 serving/                    # 🆕 Decoupled API
│   ├── api_server.py                  # FastAPI inference
│   └── inference_records/             # Audit trail
└── 📁 monitoring/                 # 🆕 MLOps utilities
    ├── drift_detector.py              # Data drift detection
    └── drift_logs.db                  # Monitoring database
```

---

## ⚠️ Technical Debt Assessment

### **Critical (Blocks 4GB VRAM target)**
- [ ] No batch size tuning for 4GB constraint
- [ ] No gradient accumulation strategy
- [ ] No memory-mapped feature loading  
- [ ] No AMP/mixed precision in all paths

### **High (Slows iteration significantly)**  
- [ ] No resumable training checkpoints mid-epoch
- [ ] No feature preprocessing pipeline
- [ ] No experiment version control
- [ ] Heavy dependencies (unused TF, CV2, etc.)

### **Medium (Quality of life)**
- [ ] No automated hyperparameter sweeps
- [ ] No model performance comparison
- [ ] No automated model card generation  
- [ ] No drift monitoring in production

---

## 🚀 Immediate Action Items

1. **Phase B**: Create `notebooks/02_feature_extract_microjobs.ipynb` + `scripts/run_feature_job.py`
2. **Phase C**: Create `notebooks/03_train_head_fastloop.ipynb` + experiment SQLite schema  
3. **Phase D**: Create `notebooks/05_seg_patch_from_features.ipynb` for patch-wise segmentation
4. **Phase E**: Create `notebooks/06_joint_finetune_safe.ipynb` with VRAM monitoring

### **Success Metrics**
- [ ] Feature extraction: 1 job processes 64 images in <2 minutes
- [ ] Head training: 3 ablations complete in <10 minutes  
- [ ] Full pipeline: Experiment iteration in <15 minutes end-to-end
- [ ] Memory usage: Stay under 4GB VRAM peak in all workflows

---

## 📋 Notebook Manifest (Current State)

| Notebook | Status | Purpose | Action |
|----------|--------|---------|---------|
| `model_training.ipynb` | 🔥 Monolith | Full TF training | **Segment into micro-jobs** |
| `colab_training.ipynb` | 🔥 Monolith | PyTorch training | **Convert to feature-cached** |  
| `eda_plant_diseases_clean.ipynb` | ✅ Good | Data exploration | Keep as reference |
| `comprehensive_*.ipynb` | ⚠️ Legacy | Old experiments | Archive/ignore |

### **New Notebooks to Create**
- `02_feature_extract_microjobs.ipynb` - Resumable feature extraction  
- `03_train_head_fastloop.ipynb` - Head-only rapid experimentation
- `05_seg_patch_from_features.ipynb` - Patch-based segmentation  
- `06_joint_finetune_safe.ipynb` - Memory-safe encoder fine-tuning
- `07_pseudo_label_and_rank.ipynb` - Active learning pipeline
- `08_distill_export.ipynb` - Student model creation + export
- `09_experiments_dashboard.ipynb` - Experiment visualization

---

**Next**: See `docs/audit_choke_points.md` for exact cell-level bottleneck analysis and micro-job segmentation strategy.