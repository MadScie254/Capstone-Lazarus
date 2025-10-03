# Capstone-Lazarus Runbook

This runbook provides step-by-step instructions for training, troubleshooting, and deploying the plant disease detection system.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Dataset Setup](#dataset-setup)
3. [Training Models](#training-models)
4. [Running the Dashboard](#running-the-dashboard)
5. [Inference & Testing](#inference--testing)
6. [Troubleshooting](#troubleshooting)
7. [Deployment](#deployment)

---

## Prerequisites

### System Requirements

- **OS**: Windows, Linux, or macOS
- **RAM**: 16 GB minimum (8 GB may work with batch_size=2-4)
- **GPU**: Optional but recommended (NVIDIA GPU with CUDA support)
- **Storage**: ~5 GB for models and checkpoints

### Software Requirements

```bash
python >= 3.8
pytorch >= 1.13
torchvision
streamlit
pillow
numpy
pandas
scikit-learn
albumentations (optional)
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Dataset Setup

### Directory Structure

The system expects data in the `./Data` directory (capital D):

```
Capstone-Lazarus/
├── Data/
│   ├── Tomato___healthy/
│   │   ├── image001.jpg
│   │   ├── image002.jpg
│   │   └── ...
│   ├── Tomato___Early_blight/
│   │   └── ...
│   ├── Corn_(maize)___healthy/
│   │   └── ...
│   └── ...
```

### Validation Checks

Before training, verify your dataset:

```python
from pathlib import Path

data_root = Path("./Data")
classes = [d.name for d in data_root.iterdir() if d.is_dir()]
print(f"Found {len(classes)} classes:")
for cls in classes:
    count = len(list((data_root / cls).glob("*.jpg"))) + len(list((data_root / cls).glob("*.png")))
    print(f"  {cls}: {count} images")
```

**Common issues:**
- Missing `./Data` folder → Create it and place your class folders inside
- Empty classes → Remove empty folders or add images
- Mixed file types → Ensure all images are .jpg or .png

---

## Training Models

### Option 1: Jupyter Notebook (Recommended for Low-Spec PCs)

Open the master trainer notebook:

```bash
jupyter notebook notebooks/master_model_trainer.ipynb
```

**For low-RAM systems:**
1. In the notebook, set `fast_test = True` for initial runs
2. Use `batch_size = 2` or `4` (instead of 8)
3. Train only the classifier head (freeze backbone)
4. Use smaller image sizes (128 or 160 instead of 224)

**Example cell modification:**

```python
# Low-memory configuration
SELECTED_MODELS = ["efficientnet_b0"]  # Train one model at a time
BATCH_SIZE = 2  # Reduce from default 8
IMAGE_SIZE = 160  # Reduce from default 224
FAST_TEST = True  # Quick sanity check first
```

Execute cells sequentially. Training should complete in 5-15 minutes for fast_test mode.

### Option 2: Python Script

Run master trainer directly:

```bash
python -m src.master_trainer --config config.yaml --fast-test
```

For full training:

```bash
python -m src.master_trainer --config config.yaml
```

### Monitoring Progress

Training logs are written to:
- Console output (live progress)
- `logs/ops.log` (full operational log)
- `models/{run_id}/run_metadata.json` (per-run metrics)

Check GPU/CPU usage:

```bash
# GPU
nvidia-smi -l 1

# CPU/RAM
htop  # Linux
resmon  # Windows
```

---

## Running the Dashboard

Launch the Streamlit console:

```bash
streamlit run app/lazarus_console/__init__.py
```

Or using the convenience alias:

```bash
python run.py
```

The dashboard will open at `http://localhost:8501`.

### Dashboard Features

**Home / Mission Readiness:**
- View latest training metrics
- See recent model checkpoints
- Quick access to inference and explainability

**Inference Laboratory:**
- Upload single images or batches
- Toggle PyTorch vs ONNX backends
- Enable ensemble mode for multi-model consensus

**Explainability Studio:**
- Generate Grad-CAM visualizations
- Adjust blend opacity
- View top-K predictions with heatmaps

**Model Comparison:**
- Compare metrics across runs
- View confusion matrices
- Analyze calibration curves

---

## Inference & Testing

### Single Image Inference

```python
from src.inference import load_model, predict_image

model = load_model("models/20251004_120000_efficientnet_b0_abc1234/best.pth")
result = predict_image(model, "path/to/image.jpg")

print(f"Prediction: {result['label']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### Batch Inference

Use the dashboard's batch upload feature or:

```python
from pathlib import Path
from src.inference import load_model, predict_batch

model = load_model("models/{run_id}/best.pth")
images = list(Path("test_images").glob("*.jpg"))
results = predict_batch(model, images)

for img, res in zip(images, results):
    print(f"{img.name}: {res['label']} ({res['confidence']:.2%})")
```

### Smoke Tests

Run the automated test suite:

```bash
# Bash (Linux/macOS/Git Bash)
./run_smoke_tests.sh

# PowerShell (Windows)
.\run_smoke_tests.ps1

# Or directly with pytest
pytest tests/test_smoke_train.py -v
pytest tests/test_streamlit_integration.py -v
```

---

## Troubleshooting

### Out of Memory (OOM) Errors

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**

1. **Reduce batch size** in `config.yaml`:
   ```yaml
   models:
     - batch_size: 2  # Down from 8
   ```

2. **Switch to CPU** (slower but uses system RAM):
   - The system automatically falls back after 3 OOM retries
   - Or force CPU mode: set `CUDA_VISIBLE_DEVICES=-1`

3. **Use smaller input sizes**:
   ```yaml
   models:
     - image_size: 128  # Down from 224
   ```

4. **Close other applications** to free RAM/VRAM

5. **Train one model at a time**:
   ```python
   SELECTED_MODELS = ["efficientnet_b0"]  # Single model
   ```

### Dataset Not Found

**Error:**
```
FileNotFoundError: Data folder './Data' not found
```

**Solution:**
- Ensure the `Data` folder exists at the repository root
- Check capitalization (must be `Data`, not `data`)
- Verify images are inside class subdirectories

### Slow Training on Laptop

**For HP ZBook 15 G5 / Quadro P2000:**

Recommended settings:
```yaml
batch_size: 4  # or 2 if still too slow
image_size: 160  # or 128
num_workers: 2  # Reduce dataloader threads
use_amp: true  # Enable mixed precision
```

Expected times (fast_test mode):
- EfficientNet-B0: 5-10 min
- MobileNetV3-Small: 3-7 min
- ResNet18: 4-8 min

### Progress Stalled

**If no epoch logs appear after 5 minutes:**

1. Check `logs/ops.log` for errors
2. Verify Data folder has readable images
3. Ensure no file corruption (try opening a few images manually)
4. Check disk space (need ~2-5 GB free)

### Model Export Failures

**ONNX export errors:**

Check `models/{run_id}/export_error.txt` for details. Common issues:
- Unsupported operations → use PyTorch checkpoint directly
- Version mismatches → update `onnx` and `onnxruntime`

**TFLite export errors:**

TFLite conversion is optional and may fail. The system will log the error and continue.

### Inference Confidence Too Low

**If predictions show < 60% confidence:**

1. Check image quality (resolution, lighting, focus)
2. Verify image matches training data distribution
3. Train longer (more epochs) or with more data
4. Use ensemble mode in the dashboard
5. Review Grad-CAM to see what the model is focusing on

---

## Deployment

### Export Model for Edge Devices

**Option 1: ONNX (recommended for edge)**

Models are automatically exported to ONNX during training. Use:

```
models/{run_id}/model.onnx
```

**Option 2: One-click package**

In the dashboard Export/Deploy section, click "📦 Package for Edge" to create a zip with:
- Model checkpoint
- Minimal inference script
- README with usage instructions
- Class names mapping

### Latency Profiling

Before deployment, profile inference speed:

```bash
python scripts/profile_inference.py \
  --model models/{run_id}/best.pth \
  --device cuda \
  --runs 100 \
  --batch-size 1
```

**Target latencies (batch=1, 224×224):**
- EfficientNet-B0 + P2000: ~40-60 ms
- MobileNetV3-Small + P2000: ~20-35 ms
- CPU (laptop i7): ~150-300 ms

### Production Checklist

- [ ] Model accuracy > 90% on validation set
- [ ] Inference latency < 500 ms (or acceptable for use case)
- [ ] Grad-CAM visualizations make sense (focuses on leaves, not background)
- [ ] Tested on representative real-world images
- [ ] Confidence thresholds calibrated (e.g., reject if < 0.7)
- [ ] Fallback UX defined (e.g., "Low confidence, please retake photo")

---

## Advanced Topics

### Resume Training from Checkpoint

The system automatically checks for `models/{run_id}/checkpoint.pth` and resumes if found.

### Custom Training Phases

Edit `config.yaml`:

```yaml
models:
  - name: efficientnet_b0
    phases:
      - type: head
        epochs: 5
        freeze_backbone: true
      - type: finetune
        epochs: 10
        freeze_backbone: false
        unfreeze_blocks: 2  # Unfreeze last 2 blocks
        learning_rate: 0.0001  # Lower LR for fine-tuning
```

### Ensemble Inference

In the dashboard, enable "🤖 Ensemble" toggle and adjust model weights. The system will:
1. Run inference on all selected models
2. Blend softmax probabilities with weighted average
3. Return consensus prediction

---

## Support & Contribution

For issues, consult:
1. This runbook
2. `logs/ops.log`
3. GitHub Issues: [Repository Issues](https://github.com/MadScie254/Capstone-Lazarus/issues)

---

**Last Updated:** October 2025  
**Version:** 1.0
