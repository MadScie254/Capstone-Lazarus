#!/usr/bin/env python3
"""
🚀 QUICK FEATURE EXTRACTION - Generate Manifest for Phase C & D

This script creates a minimal feature manifest for testing Phase C and Phase D:
1. Scans data directory for images
2. Creates manifest_features.v001.csv
3. Extracts features for small batch (32 images)
4. Enables Phase C and Phase D testing

Target: HP ZBook Quadro P2000 (4GB VRAM)
Expected Runtime: <3 minutes
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import torch
import timm
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class QuickFeatureExtractor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 8  # 4GB VRAM optimized
        self.features_dir = Path('features')
        self.data_dir = Path('data')
        
        print(f"🚀 Quick Feature Extractor")
        print(f"   Device: {self.device}")
        print(f"   Data dir: {self.data_dir}")
        print(f"   Features dir: {self.features_dir}")
        
        # Create directories
        self.features_dir.mkdir(exist_ok=True)
        
        # Initialize model
        self.model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"   ✅ EfficientNet-B0 loaded")

    def scan_dataset(self) -> pd.DataFrame:
        """Scan data directory and create image manifest"""
        print("📊 Scanning dataset...")
        
        image_records = []
        
        for class_dir in self.data_dir.iterdir():
            if not class_dir.is_dir() or class_dir.name.startswith('.'):
                continue
                
            print(f"   📁 {class_dir.name}")
            
            # Find image files
            image_files = (
                list(class_dir.glob('*.jpg')) + 
                list(class_dir.glob('*.JPG')) + 
                list(class_dir.glob('*.jpeg')) + 
                list(class_dir.glob('*.JPEG')) + 
                list(class_dir.glob('*.png')) + 
                list(class_dir.glob('*.PNG'))
            )
            
            print(f"      Found {len(image_files)} images")
            
            for img_path in image_files:
                image_id = f"{class_dir.name}_{img_path.stem}"
                image_records.append({
                    'image_id': image_id,
                    'image_path': str(img_path.absolute()),
                    'class_name': class_dir.name,
                    'file_size': img_path.stat().st_size,
                    'created_time': datetime.now().isoformat()
                })
        
        manifest_df = pd.DataFrame(image_records)
        print(f"📋 Created manifest: {len(manifest_df)} images, {manifest_df['class_name'].nunique()} classes")
        
        return manifest_df

    def extract_features_batch(self, manifest_df: pd.DataFrame, max_samples: int = 32) -> pd.DataFrame:
        """Extract features for batch of images"""
        
        # Limit to max_samples for quick testing
        test_manifest = manifest_df.head(max_samples)
        print(f"🔥 Extracting features for {len(test_manifest)} images")
        
        # Simple dataset
        class SimpleImageDataset(Dataset):
            def __init__(self, image_paths):
                self.image_paths = image_paths
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            
            def __len__(self):
                return len(self.image_paths)
            
            def __getitem__(self, idx):
                try:
                    img_path = self.image_paths[idx]
                    image = Image.open(img_path).convert('RGB')
                    return self.transform(image), img_path
                except Exception as e:
                    # Return dummy tensor if image loading fails
                    dummy = torch.zeros(3, 224, 224)
                    return dummy, self.image_paths[idx]
        
        # Create dataset and dataloader
        dataset = SimpleImageDataset(test_manifest['image_path'].tolist())
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        
        # Extract features
        all_features = []
        processed_paths = []
        
        with torch.no_grad():
            for batch_images, batch_paths in tqdm(dataloader, desc="Extracting"):
                batch_images = batch_images.to(self.device)
                features = self.model(batch_images)  # [batch_size, 1280]
                
                # Convert to float16 for memory efficiency
                features = features.cpu().numpy().astype(np.float16)
                all_features.append(features)
                processed_paths.extend(batch_paths)
        
        # Combine all features
        all_features = np.vstack(all_features)
        print(f"   ✅ Extracted features: {all_features.shape}")
        
        # Save features as NPZ cache
        encoder_dir = self.features_dir / 'encoder_efficientnet_b0'
        encoder_dir.mkdir(exist_ok=True)
        
        features_file = encoder_dir / 'batch_features.npz'
        np.savez_compressed(
            features_file,
            features=all_features,
            image_paths=processed_paths,
            extraction_time=datetime.now().isoformat()
        )
        
        print(f"   💾 Features cached: {features_file}")
        
        # Update manifest with feature files
        feature_records = []
        for i, row in test_manifest.iterrows():
            if i < len(all_features):
                feature_records.append({
                    'image_id': row['image_id'],
                    'image_path': row['image_path'],
                    'class_name': row['class_name'],
                    'feature_file': str(features_file),
                    'feature_index': i,
                    'feature_shape': str(all_features[i].shape),
                    'extraction_time': datetime.now().isoformat()
                })
        
        return pd.DataFrame(feature_records)

    def create_manifest(self, max_samples: int = 32) -> str:
        """Create complete feature manifest"""
        print("🏗️ Creating feature manifest...")
        
        # Scan dataset
        full_manifest = self.scan_dataset()
        
        if full_manifest.empty:
            print("❌ No images found in data directory")
            return None
        
        # Extract features for subset
        feature_manifest = self.extract_features_batch(full_manifest, max_samples)
        
        # Save manifest
        manifest_file = self.features_dir / 'manifest_features.v001.csv'
        feature_manifest.to_csv(manifest_file, index=False)
        
        print(f"📋 Manifest saved: {manifest_file}")
        print(f"   📊 Records: {len(feature_manifest)}")
        print(f"   🎯 Ready for Phase C & D testing")
        
        return str(manifest_file)


def main():
    """Main execution"""
    extractor = QuickFeatureExtractor()
    manifest_file = extractor.create_manifest(max_samples=32)  # Small batch for testing
    
    if manifest_file:
        print(f"\n✅ SUCCESS: Feature manifest created!")
        print(f"   📄 File: {manifest_file}")
        print(f"   🚀 You can now run Phase C and Phase D notebooks")
        return True
    else:
        print(f"\n❌ FAILED: Could not create manifest")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)