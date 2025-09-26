# 🔍 Notebook Choke-Point Analysis

**Target**: Eliminate blocking long-running cells for 4GB VRAM efficiency  
**Strategy**: Surgical micro-job segmentation with resumable checkpoints  

---

## 🎯 Primary Target: `notebooks/model_training.ipynb`

### **Cell-by-Cell Bottleneck Analysis**

| Cell # | Current Function | Choke Point | Time Estimate | Memory Usage | Action |
|--------|------------------|-------------|---------------|--------------|---------|
| **Cell 2** | Dataset loading & splitting | Heavy I/O scan | ~2-5 min | 2GB RAM | ✅ Keep (acceptable) |
| **Cell 3** | TF Dataset creation | Full image loading | ~5-10 min | 1GB RAM | 🔥 **SEGMENT** |
| **Cell 4** | Model creation | Acceptable | ~30 sec | 500MB | ✅ Keep |
| **Cell 5** | Training loop | **MAJOR CHOKE** | ~2-4 hours | 4GB+ VRAM | 🔥 **ELIMINATE** |
| **Cell 6** | Evaluation | Heavy inference | ~10-15 min | 3GB VRAM | ⚠️ Optimize |

### **Specific Problematic Code Patterns**

#### **Cell 3: Dataset Creation Bottleneck**
```python
# PROBLEM: Full image decode + augmentation per batch
def load_and_preprocess(path, label):
    image = tf.io.read_file(path)          # ❌ I/O per call
    image = tf.image.decode_image(image)   # ❌ Decode per call  
    image = tf.cast(image, tf.float32)     # ❌ Cast per call
    image = tf.image.resize(image, img_size) # ❌ Resize per call
    # ... heavy augmentation per call
```

**Solution**: Pre-extract features to `features/encoder_*/img_*.npz`

#### **Cell 5: Monolithic Training Loop** 
```python
# PROBLEM: Uninterruptible multi-hour training
history = model.fit(
    train_dataset,
    epochs=50,           # ❌ Long blocking run
    steps_per_epoch=979, # ❌ No micro-job chunking
    validation_data=validation_dataset,
    # ... callbacks but no resumable micro-jobs
)
```

**Solution**: Head-only training on cached features with job chunks

---

## 🎯 Secondary Target: `notebooks/colab_training.ipynb` 

### **PyTorch Pipeline Bottlenecks**

| Cell # | Current Function | Issue | Time | VRAM | Action |
|--------|------------------|-------|------|------|---------|
| **Cell 4** | DataLoader creation | Heavy transforms | ~1-2 min | 500MB | 🔥 **Cache features** |
| **Cell 6** | Training loop | Full training | ~1-3 hours | 3-4GB | 🔥 **Micro-job** |
| **Cell 8** | Model evaluation | Full inference | ~15-20 min | 2-3GB | ⚠️ Optimize |

### **Problematic Patterns in PyTorch Version**

#### **DataLoader with Heavy Augmentation**
```python
# PROBLEM: CPU bottleneck + repeated transforms
train_loader = DataLoader(
    dataset,
    batch_size=32,        # ❌ May OOM on 4GB
    num_workers=4,        # ❌ High CPU usage  
    # Dataset applies Albumentations per __getitem__ call
)
```

**Solution**: Pre-compute spatial features, light augmentation only

#### **Full Model Training**
```python
# PROBLEM: Long uninterruptible epoch loops
for epoch in range(config['epochs']):  # ❌ 50+ epochs
    for batch_idx, (data, target) in enumerate(train_loader):  # ❌ 979 batches
        # ... forward + backward pass
```

**Solution**: Head-only training with cached encoder features

---

## 🧩 Micro-Job Segmentation Strategy

### **Phase 1: Feature Extraction Jobs**
**Target Cell**: Replace `model_training.ipynb` Cell 3

```python
# NEW: notebooks/02_feature_extract_microjobs.ipynb
JOB_SIZE = 64  # images per job (tuned for 4GB VRAM)

def run_job(job_id, encoder_name='efficientnet_b0'):
    """Process exactly JOB_SIZE images and cache features"""
    job_manifest = pd.read_csv('features/jobs_queue.csv')
    job_row = job_manifest.iloc[job_id]
    
    image_paths = job_row['image_paths'].split(',')
    # Process batch, save to features/encoder_*/
    # Write features/job_{job_id}_{timestamp}.done
```

**Acceptance Criteria**: 
- [ ] Single job processes 64 images in <2 minutes
- [ ] Job completion marked with `.done` file  
- [ ] Features saved as float16 NPZ with global_pool

### **Phase 2: Head-Only Training Loop**
**Target Cell**: Replace `model_training.ipynb` Cell 5

```python
# NEW: notebooks/03_train_head_fastloop.ipynb  
def train_head_only(features_manifest, config):
    """Train classifier head using cached features"""
    features = load_cached_features(features_manifest)  # Memory-mapped
    
    # Create lightweight head model
    head = create_classifier_head(features.shape[1], num_classes)
    
    # Train for just 10-20 epochs (minutes, not hours)
    for epoch in range(config['head_epochs']):  # Default: 10
        # ... head-only training loop
```

**Acceptance Criteria**:
- [ ] 3 ablation configs complete in <10 minutes
- [ ] Results logged to `experiments/experiments.db`
- [ ] Checkpoints saved to `experiments/exp_*/checkpoints/`

### **Phase 3: Patch-Based Segmentation**  
**Target**: New capability for large image handling

```python
# NEW: notebooks/05_seg_patch_from_features.ipynb
def create_patch_jobs(image_manifest, patch_size=256, overlap=64):
    """Create spatial patch index for segmentation training"""
    # Generate patches/patch_manifest.v001.csv
    # Each row: patch_id, parent_image, bbox, features_path
```

**Acceptance Criteria**:
- [ ] Generate 20+ stitched sample masks
- [ ] Patch manifest includes spatial coordinates  
- [ ] Memory usage stays under 4GB during patch stitching

---

## 🎯 Specific Cell Modifications Required

### **File**: `notebooks/model_training.ipynb`

#### **Cell 3** → `notebooks/02_feature_extract_microjobs.ipynb`
- **Old**: Full TF dataset pipeline with decode/resize/augment
- **New**: Micro-job feature extraction with job queue
- **Files Created**: `features/jobs_queue.csv`, `features/encoder_*/img_*.npz`

#### **Cell 5** → `notebooks/03_train_head_fastloop.ipynb` 
- **Old**: `model.fit()` full training for 50+ epochs
- **New**: Head-only training on features for 10 epochs
- **Files Created**: `experiments/exp_*/checkpoints/best_head.pt`

#### **Cell 6** → Enhanced with cached evaluation
- **Old**: Full model inference on test set
- **New**: Fast evaluation using cached features + saved head
- **Speed**: 15 minutes → 2 minutes

### **File**: `notebooks/colab_training.ipynb`

#### **Cell 4** → Convert to feature-cached DataLoader
- **Old**: Heavy Albumentations transforms per batch
- **New**: Load pre-cached features, light augmentation only  
- **Memory**: 500MB → 200MB, 4x speed improvement

#### **Cell 6** → Head-only training variant
- **Old**: Full PyTorch model training
- **New**: Feature-cached head training with gradient accumulation
- **Time**: 1-3 hours → 10-15 minutes per experiment

---

## 🔧 Job Size Configuration (4GB VRAM Optimized)

| Operation | Batch Size | Job Size | Estimated VRAM | Time per Job |
|-----------|------------|----------|----------------|--------------|
| **Feature Extraction** | 8 | 64 images | 2.5GB | <2 min |
| **Head Training** | 128 | N/A | 1.5GB | <5 min |
| **Patch Segmentation** | 4 | 16 patches | 3.5GB | <3 min |
| **Fine-tune (last 4 blocks)** | 4 | N/A | 3.8GB | <10 min |

### **Memory Budget Allocation**
- **Model weights**: 500MB (EfficientNet-B0)
- **Feature cache**: 800MB (memory-mapped)  
- **Batch data**: 1.2GB (training batch)
- **Optimizer states**: 600MB (AdamW)
- **CUDA overhead**: 900MB (PyTorch/TF)
- **Total**: ~4.0GB (within 4GB limit)

---

## 🚨 Critical Implementation Priorities 

1. **URGENT**: `notebooks/02_feature_extract_microjobs.ipynb`
   - Replaces heaviest bottleneck (Cell 3 in model_training.ipynb)
   - Enables all downstream optimizations
   - **Acceptance**: 64 images processed in <2min, saved to NPZ

2. **HIGH**: `notebooks/03_train_head_fastloop.ipynb`
   - Enables rapid experimentation 
   - **Acceptance**: 3 head configs trained in <10min total

3. **MEDIUM**: Enhanced `notebooks/colab_training.ipynb`
   - Convert to feature-cached PyTorch pipeline
   - **Acceptance**: Training time reduced by 10x

4. **LOW**: `notebooks/05_seg_patch_from_features.ipynb`
   - New segmentation capability
   - **Acceptance**: 20 stitched masks generated successfully

---

## 📋 Implementation Checklist

### **Immediate (Phase B)**
- [ ] Create `features/` directory structure
- [ ] Implement `run_job()` function for 64-image batches
- [ ] Create job queue CSV with image path chunks
- [ ] Test single job completes in <2 minutes
- [ ] Verify NPZ features load correctly

### **Short Term (Phase C)** 
- [ ] Create SQLite schema for experiments tracking
- [ ] Implement head-only training loop
- [ ] Test 3 ablations complete in <10 minutes  
- [ ] Verify model checkpoints save correctly

### **Medium Term (Phase D-E)**
- [ ] Implement patch-based segmentation workflow
- [ ] Create safe fine-tuning with VRAM monitoring
- [ ] Test segmentation stitching produces valid masks
- [ ] Verify fine-tuning stays under 4GB VRAM

---

**Next**: Implement `notebooks/02_feature_extract_microjobs.ipynb` with the exact job-chunking strategy and NPZ caching format specified above.