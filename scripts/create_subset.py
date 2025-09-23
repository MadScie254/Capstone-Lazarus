#!/usr/bin/env python3
"""
scripts/create_subset.py
Creates a balanced stratified subset from `data/` into `out_dir` using symlinks when possible.
Usage:
  python scripts/create_subset.py --data-dir data --out-dir data_subset --samples-per-class 50 --val-ratio 0.2 --seed 42 --symlink true
"""
import argparse
from pathlib import Path
import random
import shutil
from sklearn.model_selection import train_test_split
import os
import sys

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}

def safe_symlink(src: Path, dst: Path):
    """Try to create symlink, fallback to copy if it fails"""
    try:
        os.symlink(str(src.resolve()), str(dst))
        return True
    except Exception:
        try:
            shutil.copy2(src, dst)
            return False
        except Exception as e:
            raise RuntimeError(f"Failed to link or copy {src} -> {dst}: {e}")

def collect_classes(data_dir: Path):
    """Collect all class directories from data directory"""
    classes = [p for p in sorted(data_dir.iterdir()) if p.is_dir()]
    return classes

def create_subset(data_dir: Path, out_dir: Path, samples_per_class: int = 50, val_ratio: float = 0.2, seed: int = 42, use_symlink: bool = True):
    """Create balanced stratified subset with deterministic sampling"""
    random.seed(seed)
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    if out_dir.exists():
        print(f"[create_subset] removing existing {out_dir}")
        shutil.rmtree(out_dir)
    
    (out_dir / 'train').mkdir(parents=True, exist_ok=True)
    (out_dir / 'val').mkdir(parents=True, exist_ok=True)

    classes = collect_classes(data_dir)
    print(f"[create_subset] Found {len(classes)} classes in {data_dir}")
    
    summary = {}
    total_symlinks = 0
    total_copies = 0
    
    for cls in classes:
        imgs = [p for p in cls.iterdir() if p.suffix.lower() in IMAGE_EXTS]
        if len(imgs) == 0:
            print(f"[create_subset] warning: class {cls.name} has 0 images; skipping")
            continue
            
        print(f"[create_subset] Processing {cls.name}: {len(imgs)} images available")
        
        n = min(len(imgs), samples_per_class)
        chosen = random.sample(imgs, n)
        train_imgs, val_imgs = train_test_split(chosen, test_size=val_ratio, random_state=seed)
        
        # Create class directories
        (out_dir / 'train' / cls.name).mkdir(parents=True, exist_ok=True)
        (out_dir / 'val' / cls.name).mkdir(parents=True, exist_ok=True)
        
        created_train = 0
        created_val = 0
        
        # Process training images
        for p in train_imgs:
            dst = out_dir / 'train' / cls.name / p.name
            if use_symlink:
                success = safe_symlink(p, dst)
                if success:
                    total_symlinks += 1
                else:
                    total_copies += 1
            else:
                shutil.copy2(p, dst)
                total_copies += 1
            created_train += 1
            
        # Process validation images
        for p in val_imgs:
            dst = out_dir / 'val' / cls.name / p.name
            if use_symlink:
                success = safe_symlink(p, dst)
                if success:
                    total_symlinks += 1
                else:
                    total_copies += 1
            else:
                shutil.copy2(p, dst)
                total_copies += 1
            created_val += 1
            
        summary[cls.name] = {'train': created_train, 'val': created_val}
    
    if use_symlink:
        print(f"[create_subset] Created {total_symlinks} symlinks, {total_copies} copies")
    else:
        print(f"[create_subset] Created {total_copies} copies (symlinks disabled)")
    
    return summary

def main():
    parser = argparse.ArgumentParser(description="Create balanced stratified subset for training")
    parser.add_argument('--data-dir', type=str, default='data', help='Source data directory')
    parser.add_argument('--out-dir', type=str, default='data_subset', help='Output subset directory')
    parser.add_argument('--samples-per-class', type=int, default=50, help='Max samples per class')
    parser.add_argument('--val-ratio', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--symlink', type=str, default='true', choices=['true','false'], help='Use symlinks when possible')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    use_symlink = args.symlink.lower() == 'true'
    
    print(f"[create_subset] Creating subset: {args.samples_per_class} samples per class")
    print(f"[create_subset] Validation ratio: {args.val_ratio}")
    print(f"[create_subset] Random seed: {args.seed}")
    print(f"[create_subset] Use symlinks: {use_symlink}")
    
    summary = create_subset(
        data_dir, out_dir, 
        samples_per_class=args.samples_per_class, 
        val_ratio=args.val_ratio, 
        seed=args.seed, 
        use_symlink=use_symlink
    )
    
    print("\n[create_subset] DONE. Class counts summary:")
    total_train = total_val = 0
    for k, v in summary.items():
        print(f"  {k}: train={v['train']} val={v['val']}")
        total_train += v['train']
        total_val += v['val']
    print(f"\nTotal created -> train: {total_train}, val: {total_val}")
    print(f"Subset saved to: {out_dir}")

if __name__ == '__main__':
    main()