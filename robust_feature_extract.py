#!/usr/bin/env python3
"""
🔥 ROBUST FEATURE EXTRACTION - Phase B Alternative

This script provides a robust alternative to the Phase B notebook:
1. Handles PIL/torchvision import issues gracefully
2. Creates larger feature batches for proper training
3. Works around Windows environment issues
4. Provides complete feature extraction pipeline

Target: HP ZBook Quadro P2000 (4GB VRAM)
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set environment to avoid import issues
os.environ['TIMM_FUSED_ATTN'] = '0'

class RobustFeatureExtractor:
    def __init__(self, samples_to_extract: int = 200):
        """Initialize robust feature extractor"""
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 8 if torch.cuda.is_available() else 4
        self.samples_to_extract = samples_to_extract
        
        # Paths
        self.data_dir = Path('data')
        self.features_dir = Path('features')
        self.encoder_dir = self.features_dir / 'encoder_efficientnet_b0'
        
        print(f"🔥 ROBUST FEATURE EXTRACTOR")
        print(f"   Device: {self.device}")
        print(f"   Target samples: {samples_to_extract}")
        print(f"   Batch size: {self.batch_size}")
        
        # Create directories
        self.features_dir.mkdir(exist_ok=True)
        self.encoder_dir.mkdir(exist_ok=True)
        
        # Initialize model with error handling
        self.model = self._init_model()
        
        # Initialize transforms with fallback
        self.transform = self._init_transforms()
        
    def _init_model(self):
        """Initialize model with error handling"""
        try:
            import timm
            model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
            model = model.to(self.device)
            model.eval()
            print(f"   ✅ EfficientNet-B0 loaded via TIMM")
            return model
            
        except Exception as e:
            print(f"   ⚠️ TIMM failed: {e}")
            print(f"   🔄 Trying PyTorch Hub fallback...")
            
            try:
                # Fallback to torch hub
                model = torch.hub.load('pytorch/vision:v0.10.0', 'efficientnet_b0', pretrained=True)
                
                # Remove classifier for feature extraction
                if hasattr(model, 'classifier'):
                    model.classifier = nn.Identity()
                elif hasattr(model, 'fc'):
                    model.fc = nn.Identity()
                
                model = model.to(self.device)
                model.eval()
                print(f"   ✅ EfficientNet-B0 loaded via PyTorch Hub")
                return model
                
            except Exception as e2:
                print(f"   ❌ All model loading failed: {e2}")
                raise RuntimeError("Could not load any feature extraction model")
    
    def _init_transforms(self):
        """Initialize transforms with fallback"""
        try:
            import torchvision.transforms as T
            transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            print(f"   ✅ Torchvision transforms loaded")
            return transform
            
        except ImportError:
            print(f"   ⚠️ Torchvision transforms failed - using manual transforms")
            return self._manual_transform
    
    def _manual_transform(self, image):
        """Manual transform fallback"""
        from PIL import Image
        import torch
        
        # Resize
        image = image.resize((224, 224))
        
        # To tensor
        tensor = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
        
        # Normalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std
        
        return tensor
    
    def scan_and_select_images(self) -> List[Tuple[str, str]]:
        """Scan dataset and select images for extraction"""
        print("📊 Scanning dataset for balanced sampling...")
        
        class_images = {}
        
        # Collect images by class
        for class_dir in self.data_dir.iterdir():
            if not class_dir.is_dir() or class_dir.name.startswith('.'):
                continue
                
            images = []
            for ext in ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']:
                images.extend(class_dir.glob(ext))
            
            if images:
                class_images[class_dir.name] = images
                print(f"   📁 {class_dir.name}: {len(images)} images")
        
        # Select balanced samples
        samples_per_class = max(1, self.samples_to_extract // len(class_images))
        selected_images = []
        
        for class_name, images in class_images.items():
            # Take up to samples_per_class from each class
            selected = images[:min(samples_per_class, len(images))]
            for img_path in selected:
                selected_images.append((str(img_path), class_name))
        
        # If we need more samples, take additional ones
        if len(selected_images) < self.samples_to_extract:
            remaining = self.samples_to_extract - len(selected_images)
            all_remaining = []
            
            for class_name, images in class_images.items():
                for img_path in images[samples_per_class:]:
                    all_remaining.append((str(img_path), class_name))
            
            # Add remaining samples
            selected_images.extend(all_remaining[:remaining])
        
        print(f"📋 Selected {len(selected_images)} images from {len(class_images)} classes")
        return selected_images[:self.samples_to_extract]
    
    def extract_features_robust(self, image_list: List[Tuple[str, str]]) -> pd.DataFrame:
        """Extract features with robust error handling"""
        print(f"🔥 Extracting features for {len(image_list)} images...")
        
        feature_records = []
        successful_extractions = 0
        
        # Process in batches
        for i in tqdm(range(0, len(image_list), self.batch_size), desc="Processing"):
            batch_images = image_list[i:i + self.batch_size]
            
            try:
                # Load and process batch
                batch_tensors = []
                batch_info = []
                
                for img_path, class_name in batch_images:
                    try:
                        # Load image
                        from PIL import Image
                        image = Image.open(img_path).convert('RGB')
                        
                        # Transform
                        if callable(self.transform):
                            if hasattr(self.transform, 'transforms'):  # torchvision
                                tensor = self.transform(image)
                            else:  # manual transform
                                tensor = self.transform(image)
                        else:
                            # Last resort manual transform
                            tensor = torch.tensor(np.array(image.resize((224, 224)))).permute(2, 0, 1).float() / 255.0
                        
                        batch_tensors.append(tensor)
                        batch_info.append((img_path, class_name))
                        
                    except Exception as e:
                        print(f"      ⚠️ Failed to load {Path(img_path).name}: {e}")
                        continue
                
                if not batch_tensors:
                    continue
                
                # Stack tensors and move to device
                batch_tensor = torch.stack(batch_tensors).to(self.device)
                
                # Extract features
                with torch.no_grad():
                    features = self.model(batch_tensor)
                    features = features.cpu().numpy().astype(np.float16)
                
                # Save individual feature files
                for j, (feature_vec, (img_path, class_name)) in enumerate(zip(features, batch_info)):
                    img_path_obj = Path(img_path)
                    image_id = f"{class_name}_{img_path_obj.stem}"
                    
                    # Create unique filename
                    feature_filename = f"img_{successful_extractions:04d}_{image_id}.npz"
                    feature_file = self.encoder_dir / feature_filename
                    
                    # Save feature file
                    np.savez_compressed(
                        feature_file,
                        features=feature_vec,
                        image_path=img_path,
                        image_id=image_id,
                        class_name=class_name,
                        extraction_time=datetime.now().isoformat()
                    )
                    
                    # Add to manifest
                    feature_records.append({
                        'image_id': image_id,
                        'image_path': img_path,
                        'class_name': class_name,
                        'feature_file': f"../features/encoder_efficientnet_b0/{feature_filename}",
                        'feature_shape': str(feature_vec.shape),
                        'extraction_time': datetime.now().isoformat()
                    })
                    
                    successful_extractions += 1
                
                # Clear GPU memory periodically
                if torch.cuda.is_available() and i % 50 == 0:
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"   ⚠️ Batch {i//self.batch_size + 1} failed: {e}")
                continue
        
        print(f"✅ Successfully extracted {successful_extractions} features")
        return pd.DataFrame(feature_records)
    
    def create_robust_manifest(self) -> str:
        """Create robust feature manifest"""
        print("🏗️ Creating robust feature manifest...")
        
        # Select images
        image_list = self.scan_and_select_images()
        
        if not image_list:
            raise ValueError("No images found for feature extraction")
        
        # Extract features
        manifest_df = self.extract_features_robust(image_list)
        
        if manifest_df.empty:
            raise ValueError("No features extracted successfully")
        
        # Save manifest
        manifest_file = self.features_dir / 'manifest_features.v001.csv'
        manifest_df.to_csv(manifest_file, index=False)
        
        print(f"📋 Manifest saved: {manifest_file}")
        print(f"   📊 Records: {len(manifest_df)}")
        print(f"   🎯 Classes: {manifest_df['class_name'].nunique()}")
        print(f"   📁 Feature files in: {self.encoder_dir}")
        
        return str(manifest_file)


def main():
    """Main execution"""
    print("🔥 STARTING ROBUST FEATURE EXTRACTION")
    print("=" * 60)
    
    try:
        # Create extractor with more samples for proper training
        extractor = RobustFeatureExtractor(samples_to_extract=200)
        
        # Create manifest
        manifest_file = extractor.create_robust_manifest()
        
        print(f"\n✅ SUCCESS!")
        print(f"   📄 Manifest: {manifest_file}")
        print(f"   🚀 Phase C and D can now run with proper dataset size")
        print(f"   🎯 Ready for head training and patch segmentation")
        
        return True
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)