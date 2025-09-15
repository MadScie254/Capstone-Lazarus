#!/usr/bin/env python3
"""
Model Size Validation Script for CAPSTONE-LAZARUS
================================================

This script checks model sizes to prevent accidentally committing
large model files to the repository.
"""

import sys
import os
from pathlib import Path
import json
from typing import List, Dict, Any

# Configuration
MAX_MODEL_SIZE_MB = 200  # Maximum model file size in MB
MODEL_EXTENSIONS = {'.h5', '.keras', '.pkl', '.joblib', '.onnx', '.pb', '.pt', '.pth'}
ALLOWED_SMALL_MODELS = {'metadata.json', 'config.json', 'model_config.yaml'}

def check_model_size(file_path: Path) -> tuple[bool, float]:
    """Check model file size"""
    try:
        size_mb = file_path.stat().st_size / (1024 * 1024)
        return size_mb <= MAX_MODEL_SIZE_MB, size_mb
    except Exception as e:
        print(f"❌ ERROR: Cannot check file size for {file_path}: {e}")
        return False, 0

def is_model_file(file_path: Path) -> bool:
    """Check if file is a model file"""
    return (
        file_path.suffix.lower() in MODEL_EXTENSIONS or
        'model' in file_path.name.lower() or
        file_path.name in ALLOWED_SMALL_MODELS
    )

def get_model_info(file_path: Path) -> Dict[str, Any]:
    """Extract model information if possible"""
    info = {
        "type": "unknown",
        "size_mb": 0,
        "parameters": None,
        "framework": "unknown"
    }
    
    try:
        size_mb = file_path.stat().st_size / (1024 * 1024)
        info["size_mb"] = round(size_mb, 2)
        
        # Detect framework and type
        if file_path.suffix.lower() in ['.h5', '.keras']:
            info["framework"] = "tensorflow"
            info["type"] = "keras_model"
        elif file_path.suffix.lower() in ['.pt', '.pth']:
            info["framework"] = "pytorch"
            info["type"] = "pytorch_model"
        elif file_path.suffix.lower() == '.onnx':
            info["framework"] = "onnx"
            info["type"] = "onnx_model"
        elif file_path.suffix.lower() in ['.pkl', '.joblib']:
            info["framework"] = "sklearn"
            info["type"] = "sklearn_model"
        elif file_path.suffix.lower() == '.pb':
            info["framework"] = "tensorflow"
            info["type"] = "tensorflow_saved_model"
        
        # Try to get parameter count for specific formats
        if file_path.suffix.lower() in ['.h5', '.keras'] and size_mb < 50:  # Only for smaller files
            try:
                import tensorflow as tf
                model = tf.keras.models.load_model(file_path, compile=False)
                info["parameters"] = model.count_params()
                del model  # Free memory
            except Exception:
                pass
                
    except Exception as e:
        print(f"⚠️  Warning: Cannot analyze model {file_path}: {e}")
    
    return info

def suggest_alternatives(file_path: Path, size_mb: float) -> List[str]:
    """Suggest alternatives for large model files"""
    suggestions = []
    
    if size_mb > MAX_MODEL_SIZE_MB:
        suggestions.extend([
            f"Move {file_path.name} to external storage (AWS S3, Google Drive, etc.)",
            "Use Git LFS for large files",
            "Upload to model registry (MLflow, Hugging Face Hub)",
            "Consider model compression techniques",
            "Store only model metadata and download link"
        ])
    
    if file_path.suffix.lower() in ['.h5', '.keras']:
        suggestions.extend([
            "Save model architecture and weights separately",
            "Use model.save_weights() instead of model.save()",
            "Consider using TensorFlow Lite for mobile deployment"
        ])
    
    if file_path.suffix.lower() in ['.pkl', '.joblib']:
        suggestions.extend([
            "Consider using joblib with compression",
            "Save only essential model parameters",
            "Use model serialization formats like ONNX"
        ])
    
    return suggestions

def create_model_registry(model_files: List[Path]) -> bool:
    """Create/update model registry"""
    registry_path = Path("models/.model_registry.json")
    
    registry = {
        "version": "1.0",
        "updated_at": "",
        "models": {}
    }
    
    # Load existing registry
    if registry_path.exists():
        try:
            with open(registry_path, 'r') as f:
                registry = json.load(f)
        except Exception:
            pass
    
    # Update registry
    from datetime import datetime
    registry["updated_at"] = datetime.now().isoformat()
    
    for file_path in model_files:
        if file_path.name == '.model_registry.json':
            continue
            
        relative_path = str(file_path.relative_to(Path("models")))
        model_info = get_model_info(file_path)
        
        # Add additional metadata
        model_info.update({
            "path": relative_path,
            "created_at": datetime.fromtimestamp(file_path.stat().st_ctime).isoformat(),
            "modified_at": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
        })
        
        registry["models"][relative_path] = model_info
    
    # Write registry
    try:
        registry_path.parent.mkdir(exist_ok=True)
        with open(registry_path, 'w') as f:
            json.dump(registry, f, indent=2)
        return True
    except Exception as e:
        print(f"❌ ERROR: Cannot update model registry: {e}")
        return False

def main():
    """Main model size validation function"""
    print("🔍 Running model size validation...")
    
    # Find model files being committed
    model_files = []
    
    try:
        import subprocess
        result = subprocess.run(
            ['git', 'diff', '--cached', '--name-only'],
            capture_output=True,
            text=True,
            cwd=os.getcwd()
        )
        
        if result.returncode == 0:
            staged_files = result.stdout.strip().split('\n')
            model_files = [
                Path(f) for f in staged_files 
                if Path(f).exists() and is_model_file(Path(f))
            ]
        else:
            # Fallback: check models directory
            models_dir = Path('models')
            if models_dir.exists():
                model_files = [
                    f for f in models_dir.rglob('*') 
                    if f.is_file() and is_model_file(f)
                ]
    
    except Exception:
        # Fallback: check models directory
        models_dir = Path('models')
        if models_dir.exists():
            model_files = [
                f for f in models_dir.rglob('*') 
                if f.is_file() and is_model_file(f)
            ]
    
    if not model_files:
        print("✅ No model files to check")
        return 0
    
    print(f"📊 Checking {len(model_files)} model files...")
    
    errors = []
    warnings = []
    large_models = []
    
    for file_path in model_files:
        is_valid, size_mb = check_model_size(file_path)
        
        if not is_valid:
            errors.append(f"Model file too large: {file_path} ({size_mb:.1f}MB > {MAX_MODEL_SIZE_MB}MB)")
            large_models.append((file_path, size_mb))
        elif size_mb > 50:  # Warning for medium-sized files
            warnings.append(f"Large model file: {file_path} ({size_mb:.1f}MB)")
        
        # Additional checks
        model_info = get_model_info(file_path)
        if model_info["framework"] == "unknown" and size_mb > 10:
            warnings.append(f"Unknown model format: {file_path}")
    
    # Update model registry
    if model_files:
        print("📝 Updating model registry...")
        if not create_model_registry(model_files):
            warnings.append("Failed to update model registry")
    
    # Report results
    if warnings:
        print("\n⚠️  WARNINGS:")
        for warning in warnings:
            print(f"  {warning}")
    
    if errors:
        print("\n❌ ERRORS:")
        for error in errors:
            print(f"  {error}")
        
        print("\n💡 SUGGESTIONS:")
        for file_path, size_mb in large_models:
            print(f"\n  For {file_path.name} ({size_mb:.1f}MB):")
            suggestions = suggest_alternatives(file_path, size_mb)
            for suggestion in suggestions[:3]:  # Show top 3 suggestions
                print(f"    • {suggestion}")
        
        print("\n📋 GENERAL RECOMMENDATIONS:")
        print("  • Use model versioning systems (MLflow, DVC)")
        print("  • Store models in cloud storage with metadata")
        print("  • Implement model compression techniques")
        print("  • Use Git LFS for unavoidable large files")
        
        return 1
    
    print("✅ All model size checks passed!")
    
    # Summary
    if model_files:
        total_size = sum(get_model_info(f)["size_mb"] for f in model_files)
        print(f"📈 Total model size: {total_size:.1f}MB across {len(model_files)} files")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())