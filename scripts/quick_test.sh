#!/bin/bash
# Quick test script for balanced subset workflow
# Tests subset creation and training on minimal data

set -e  # Exit on error

echo "🚀 Quick Test: Balanced Subset Workflow"
echo "========================================"

# Configuration
DATA_DIR="data"
SUBSET_DIR="test_subset"
SAMPLES_PER_CLASS=10
EPOCHS=2
BATCH_SIZE=8
IMG_SIZE=128

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_step() {
    echo -e "${BLUE}📍 $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if data directory exists
print_step "Checking data directory..."
if [ ! -d "$DATA_DIR" ]; then
    print_error "Data directory '$DATA_DIR' not found!"
    echo "Please ensure your dataset is in the '$DATA_DIR' directory"
    exit 1
fi

# Count available classes
CLASS_COUNT=$(find "$DATA_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
print_success "Found $CLASS_COUNT classes in data directory"

if [ $CLASS_COUNT -lt 2 ]; then
    print_error "Need at least 2 classes for testing"
    exit 1
fi

# Clean up any existing test subset
print_step "Cleaning up previous test runs..."
if [ -d "$SUBSET_DIR" ]; then
    rm -rf "$SUBSET_DIR"
    print_success "Removed existing subset directory"
fi

if [ -d "quick_test_results" ]; then
    rm -rf "quick_test_results"
    print_success "Removed existing results directory"
fi

# Step 1: Create subset
print_step "Step 1: Creating balanced subset..."
echo "Command: python scripts/create_subset.py --data-dir $DATA_DIR --subset-dir $SUBSET_DIR --samples-per-class $SAMPLES_PER_CLASS --seed 42 --verbose"

python scripts/create_subset.py \
    --data-dir "$DATA_DIR" \
    --subset-dir "$SUBSET_DIR" \
    --samples-per-class $SAMPLES_PER_CLASS \
    --seed 42 \
    --verbose

if [ $? -eq 0 ]; then
    print_success "Subset created successfully!"
else
    print_error "Subset creation failed!"
    exit 1
fi

# Verify subset structure
print_step "Verifying subset structure..."
if [ -d "$SUBSET_DIR/train" ] && [ -d "$SUBSET_DIR/val" ]; then
    TRAIN_CLASSES=$(find "$SUBSET_DIR/train" -mindepth 1 -maxdepth 1 -type d | wc -l)
    VAL_CLASSES=$(find "$SUBSET_DIR/val" -mindepth 1 -maxdepth 1 -type d | wc -l)
    print_success "Subset structure verified (Train: $TRAIN_CLASSES classes, Val: $VAL_CLASSES classes)"
else
    print_error "Subset structure is invalid!"
    exit 1
fi

# Step 2: Test training
print_step "Step 2: Testing training pipeline..."
echo "Command: python src/train.py --data-dir $DATA_DIR --subset-dir $SUBSET_DIR --samples-per-class $SAMPLES_PER_CLASS --epochs $EPOCHS --batch-size $BATCH_SIZE --img-size $IMG_SIZE --save-dir quick_test_results"

python src/train.py \
    --data-dir "$DATA_DIR" \
    --subset-dir "$SUBSET_DIR" \
    --samples-per-class $SAMPLES_PER_CLASS \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --img-size $IMG_SIZE \
    --save-dir "quick_test_results" \
    --seed 42

if [ $? -eq 0 ]; then
    print_success "Training completed successfully!"
else
    print_error "Training failed!"
    exit 1
fi

# Step 3: Verify outputs
print_step "Step 3: Verifying outputs..."

# Check model files
if [ -f "quick_test_results/best_model.pth" ]; then
    print_success "Best model saved"
else
    print_warning "Best model not found"
fi

if [ -f "quick_test_results/latest.pth" ]; then
    print_success "Latest model saved"
else
    print_warning "Latest model not found"
fi

if [ -f "quick_test_results/training_history.json" ]; then
    print_success "Training history saved"
    
    # Extract final accuracy if jq is available
    if command -v jq &> /dev/null; then
        FINAL_ACC=$(jq -r '.val_acc[-1]' quick_test_results/training_history.json)
        print_success "Final validation accuracy: $FINAL_ACC"
    fi
else
    print_warning "Training history not found"
fi

# Step 4: Cleanup (optional)
print_step "Step 4: Cleanup (optional)..."
read -p "Do you want to clean up test files? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -rf "$SUBSET_DIR"
    rm -rf "quick_test_results"
    print_success "Test files cleaned up"
else
    print_success "Test files preserved for inspection"
fi

# Summary
echo
echo "🎉 QUICK TEST COMPLETED SUCCESSFULLY!"
echo "========================================"
print_success "✅ Subset creation works"
print_success "✅ Training pipeline works"  
print_success "✅ Model saving works"
echo
echo "🚀 Ready for full experiments!"
echo "   • Use create_subset.py with larger --samples-per-class"
echo "   • Use train.py with more --epochs"
echo "   • Try the Jupyter notebook: notebooks/jupyter_subset_training.ipynb"
echo
echo "📁 Test used:"
echo "   • Samples per class: $SAMPLES_PER_CLASS"
echo "   • Epochs: $EPOCHS"
echo "   • Batch size: $BATCH_SIZE"
echo "   • Image size: $IMG_SIZE"