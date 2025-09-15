#!/usr/bin/env python3
"""
Data Protection Script for CAPSTONE-LAZARUS
==========================================

This script prevents accidental commits of sensitive data or large datasets.
Part of the pre-commit hooks to ensure data safety.
"""

import sys
import os
from pathlib import Path
import hashlib
import json
from typing import List, Dict, Any

# Configuration
MAX_FILE_SIZE_MB = 100  # Maximum file size in MB
ALLOWED_EXTENSIONS = {'.txt', '.md', '.json', '.yaml', '.yml', '.gitkeep'}
PROTECTED_PATTERNS = [
    '*.jpg', '*.jpeg', '*.png', '*.tiff', '*.bmp',  # Images
    '*.mp4', '*.avi', '*.mov', '*.mkv',  # Videos
    '*.zip', '*.tar', '*.gz', '*.rar',   # Archives
    '*.h5', '*.pkl', '*.joblib', '*.onnx'  # Large model files
]

def check_file_size(file_path: Path) -> bool:
    """Check if file size is within limits"""
    try:
        size_mb = file_path.stat().st_size / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            print(f"❌ ERROR: File too large: {file_path} ({size_mb:.1f}MB > {MAX_FILE_SIZE_MB}MB)")
            return False
        return True
    except Exception as e:
        print(f"❌ ERROR: Cannot check file size for {file_path}: {e}")
        return False

def is_protected_pattern(file_path: Path) -> bool:
    """Check if file matches protected patterns"""
    file_name = file_path.name.lower()
    
    # Check against protected patterns
    import fnmatch
    for pattern in PROTECTED_PATTERNS:
        if fnmatch.fnmatch(file_name, pattern.lower()):
            return True
    
    return False

def is_allowed_extension(file_path: Path) -> bool:
    """Check if file extension is allowed in data directory"""
    return file_path.suffix.lower() in ALLOWED_EXTENSIONS

def check_data_integrity(file_path: Path) -> bool:
    """Verify data file integrity and metadata"""
    if not file_path.exists():
        return True
    
    # Check for common data corruption indicators
    if file_path.suffix.lower() in ['.json', '.yaml', '.yml']:
        try:
            if file_path.suffix.lower() == '.json':
                with open(file_path, 'r') as f:
                    json.load(f)
            elif file_path.suffix.lower() in ['.yaml', '.yml']:
                import yaml
                with open(file_path, 'r') as f:
                    yaml.safe_load(f)
        except Exception as e:
            print(f"❌ ERROR: Invalid data file format {file_path}: {e}")
            return False
    
    return True

def generate_file_hash(file_path: Path) -> str:
    """Generate SHA256 hash for file"""
    hash_sha256 = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    except Exception:
        return ""

def update_data_manifest(data_files: List[Path]) -> bool:
    """Update data manifest with file information"""
    manifest_path = Path("data/.data_manifest.json")
    
    manifest = {
        "version": "1.0",
        "generated_at": "",
        "files": {}
    }
    
    # Load existing manifest if it exists
    if manifest_path.exists():
        try:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
        except Exception:
            pass  # Use default manifest
    
    # Update manifest
    from datetime import datetime
    manifest["generated_at"] = datetime.now().isoformat()
    
    for file_path in data_files:
        if file_path.name == '.data_manifest.json':
            continue
            
        relative_path = str(file_path.relative_to(Path("data")))
        file_info = {
            "size_bytes": file_path.stat().st_size,
            "modified_at": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
            "hash": generate_file_hash(file_path),
            "type": file_path.suffix.lower()
        }
        
        manifest["files"][relative_path] = file_info
    
    # Write updated manifest
    try:
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        return True
    except Exception as e:
        print(f"❌ ERROR: Cannot update data manifest: {e}")
        return False

def main():
    """Main data protection function"""
    print("🛡️  Running data protection checks...")
    
    # Get files being committed in data directory
    data_files = []
    
    # Check if we're in a git repository and get staged files
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
            data_files = [
                Path(f) for f in staged_files 
                if f.startswith('data/') and Path(f).exists()
            ]
        else:
            # Fallback: check all files in data directory
            data_dir = Path('data')
            if data_dir.exists():
                data_files = [
                    f for f in data_dir.rglob('*') 
                    if f.is_file() and not f.name.startswith('.')
                ]
    
    except Exception:
        # Fallback: check all files in data directory
        data_dir = Path('data')
        if data_dir.exists():
            data_files = [
                f for f in data_dir.rglob('*') 
                if f.is_file() and not f.name.startswith('.')
            ]
    
    if not data_files:
        print("✅ No data files to check")
        return 0
    
    print(f"🔍 Checking {len(data_files)} data files...")
    
    errors = []
    warnings = []
    
    for file_path in data_files:
        # Check file size
        if not check_file_size(file_path):
            errors.append(f"File too large: {file_path}")
        
        # Check protected patterns
        if is_protected_pattern(file_path):
            warnings.append(f"Protected file pattern detected: {file_path}")
            
            # For protected patterns, check if allowed extension
            if not is_allowed_extension(file_path):
                errors.append(f"Large binary file in data directory: {file_path}")
        
        # Check data integrity
        if not check_data_integrity(file_path):
            errors.append(f"Data integrity check failed: {file_path}")
    
    # Update data manifest
    if data_files:
        print("📝 Updating data manifest...")
        if not update_data_manifest(data_files):
            warnings.append("Failed to update data manifest")
    
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
        print("  • Move large files to external storage (cloud, LFS)")
        print("  • Add file patterns to .gitignore")
        print("  • Use data versioning tools like DVC")
        print("  • Compress files if necessary")
        print("  • Ensure only metadata files are committed")
        
        return 1
    
    print("✅ All data protection checks passed!")
    return 0

if __name__ == "__main__":
    sys.exit(main())