# Smoke test runner for Capstone-Lazarus training pipeline (PowerShell)

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Running Capstone-Lazarus Smoke Tests" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

try {
    # Run training smoke tests
    Write-Host ""
    Write-Host "→ Testing master trainer pipeline..." -ForegroundColor Yellow
    pytest tests/test_smoke_train.py -v --tb=short
    if ($LASTEXITCODE -ne 0) { throw "Training smoke tests failed" }

    # Run Streamlit integration tests
    Write-Host ""
    Write-Host "→ Testing Streamlit dashboard integration..." -ForegroundColor Yellow
    pytest tests/test_streamlit_integration.py -v --tb=short
    if ($LASTEXITCODE -ne 0) { throw "Streamlit integration tests failed" }

    # Run master trainer unit tests
    Write-Host ""
    Write-Host "→ Testing master trainer unit tests..." -ForegroundColor Yellow
    pytest tests/test_master_trainer.py -v --tb=short
    if ($LASTEXITCODE -ne 0) { throw "Master trainer tests failed" }

    Write-Host ""
    Write-Host "==========================================" -ForegroundColor Green
    Write-Host "✓ All smoke tests passed!" -ForegroundColor Green
    Write-Host "==========================================" -ForegroundColor Green
    
} catch {
    Write-Host ""
    Write-Host "==========================================" -ForegroundColor Red
    Write-Host "✗ Smoke tests failed: $_" -ForegroundColor Red
    Write-Host "==========================================" -ForegroundColor Red
    exit 1
}
