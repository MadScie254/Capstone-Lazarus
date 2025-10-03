#!/usr/bin/env bash
# Smoke test runner for Capstone-Lazarus training pipeline

set -e

echo "=========================================="
echo "Running Capstone-Lazarus Smoke Tests"
echo "=========================================="

# Run training smoke tests
echo ""
echo "→ Testing master trainer pipeline..."
pytest tests/test_smoke_train.py -v --tb=short

# Run Streamlit integration tests
echo ""
echo "→ Testing Streamlit dashboard integration..."
pytest tests/test_streamlit_integration.py -v --tb=short

# Run master trainer unit tests
echo ""
echo "→ Testing master trainer unit tests..."
pytest tests/test_master_trainer.py -v --tb=short

echo ""
echo "=========================================="
echo "✓ All smoke tests passed!"
echo "=========================================="
