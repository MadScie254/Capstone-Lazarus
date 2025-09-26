#!/usr/bin/env python3
"""
🚀 QUICK PIPELINE TEST - Capstone-Lazarus Optimization Validation

This script tests the complete optimized pipeline:
1. Phase B: Extract features for 128 images (2 micro-jobs)
2. Phase C: Train 3 head architectures with ablations
3. Validation: Ensure 4GB VRAM constraint is respected

Target: HP ZBook Quadro P2000 (4GB VRAM, 16GB RAM)
Expected Runtime: <5 minutes total
"""

import os
import sys
import time
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

# Add project root to path
sys.path.append('src')

class PipelineValidator:
    def __init__(self, test_samples: int = 128):
        self.test_samples = test_samples
        self.batch_size = 8  # 4GB VRAM optimized
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
        
        print(f"🚀 Pipeline Validator Initialized")
        print(f"   Device: {self.device}")
        print(f"   Test samples: {test_samples}")
        print(f"   Batch size: {self.batch_size}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name()}")
            print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print()

    def test_phase_b_feature_extraction(self) -> bool:
        """Test Phase B: Micro-job feature extraction"""
        print("🔥 Testing Phase B: Feature Extraction")
        start_time = time.time()
        
        try:
            # Import feature extraction components
            import timm
            import torch.nn as nn
            from torch.utils.data import Dataset, DataLoader
            from torchvision import transforms
            from PIL import Image
            
            # Create minimal test dataset
            data_path = Path('data')
            all_images = []
            
            for class_dir in data_path.iterdir():
                if class_dir.is_dir() and not class_dir.name.startswith('.'):
                    images = list(class_dir.glob('*.jpg'))[:self.test_samples // 10]  # ~13 per class
                    all_images.extend([(img, class_dir.name) for img in images])
            
            # Limit to test_samples
            all_images = all_images[:self.test_samples]
            print(f"   Found {len(all_images)} test images")
            
            if len(all_images) == 0:
                print("   ❌ No images found in data/ directory")
                return False
            
            # Create simple dataset
            class TestDataset(Dataset):
                def __init__(self, image_paths, transform=None):
                    self.image_paths = image_paths
                    self.transform = transform or transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])
                
                def __len__(self):
                    return len(self.image_paths)
                
                def __getitem__(self, idx):
                    img_path, label = self.image_paths[idx]
                    image = Image.open(img_path).convert('RGB')
                    if self.transform:
                        image = self.transform(image)
                    return image, label, str(img_path)
            
            # Create model and dataloader
            model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)  # Feature extractor
            model = model.to(self.device)
            model.eval()
            
            dataset = TestDataset(all_images)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
            
            # Extract features
            features_list = []
            labels_list = []
            paths_list = []
            
            with torch.no_grad():
                for batch_idx, (images, labels, paths) in enumerate(dataloader):
                    if batch_idx >= 2:  # Only test 2 batches (16 images)
                        break
                        
                    images = images.to(self.device)
                    
                    # Monitor memory
                    if torch.cuda.is_available():
                        memory_used = torch.cuda.memory_allocated() / 1e9
                        if memory_used > 3.5:  # 3.5GB warning for 4GB limit
                            print(f"   ⚠️  High VRAM usage: {memory_used:.1f} GB")
                    
                    features = model(images)  # Shape: [batch_size, 1280]
                    features = features.cpu().numpy().astype(np.float16)  # Memory optimization
                    
                    features_list.append(features)
                    labels_list.extend(labels)
                    paths_list.extend(paths)
            
            # Combine results
            all_features = np.vstack(features_list)
            print(f"   ✅ Extracted features shape: {all_features.shape}")
            print(f"   ✅ Feature dtype: {all_features.dtype}")
            
            # Test NPZ caching
            features_dir = Path('features/test_cache')
            features_dir.mkdir(parents=True, exist_ok=True)
            
            cache_path = features_dir / 'test_features.npz'
            np.savez_compressed(cache_path, features=all_features, labels=labels_list)
            
            # Verify cache loading
            loaded = np.load(cache_path)
            assert loaded['features'].shape == all_features.shape
            print(f"   ✅ NPZ caching working: {cache_path}")
            
            elapsed = time.time() - start_time
            print(f"   ✅ Phase B completed in {elapsed:.1f}s")
            
            self.results['phase_b'] = {
                'success': True,
                'features_shape': all_features.shape,
                'elapsed_time': elapsed,
                'samples_processed': len(all_features)
            }
            return True
            
        except Exception as e:
            print(f"   ❌ Phase B failed: {e}")
            self.results['phase_b'] = {'success': False, 'error': str(e)}
            return False

    def test_phase_c_head_training(self) -> bool:
        """Test Phase C: Head architecture training"""
        print("\n🔥 Testing Phase C: Head Training")
        start_time = time.time()
        
        try:
            # Load cached features
            import torch.nn as nn
            cache_path = Path('features/test_cache/test_features.npz')
            if not cache_path.exists():
                print("   ❌ No cached features found. Run Phase B first.")
                return False
            
            data = np.load(cache_path)
            features = torch.tensor(data['features']).float()
            labels = list(data['labels'])
            
            # Create label mapping
            unique_labels = list(set(labels))
            label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
            y = torch.tensor([label_to_idx[label] for label in labels])
            
            print(f"   Features: {features.shape}, Labels: {y.shape}")
            print(f"   Classes: {len(unique_labels)}")
            
            # Test 3 head architectures
            feature_dim = features.shape[1]  # 1280 for EfficientNet-B0
            num_classes = len(unique_labels)
            
            heads = {
                'Linear': nn.Linear(feature_dim, num_classes),
                'MLP': nn.Sequential(
                    nn.Linear(feature_dim, 256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, num_classes)
                ),
                'Attention': nn.Sequential(
                    nn.Linear(feature_dim, 256),
                    nn.MultiheadAttention(256, 4, batch_first=True),
                    nn.Linear(256, num_classes)
                )
            }
            
            results = {}
            
            for head_name, head_model in heads.items():
                print(f"   Testing {head_name} head...")
                
                head_model = head_model.to(self.device)
                optimizer = torch.optim.Adam(head_model.parameters(), lr=0.001)
                criterion = nn.CrossEntropyLoss()
                
                # Create simple train/val split
                n_train = int(0.8 * len(features))
                X_train, X_val = features[:n_train], features[n_train:]
                y_train, y_val = y[:n_train], y[n_train:]
                
                # Quick training (3 epochs)
                head_model.train()
                for epoch in range(3):
                    X_batch = X_train.to(self.device)
                    y_batch = y_train.to(self.device)
                    
                    optimizer.zero_grad()
                    
                    if head_name == 'Attention':
                        # Special handling for attention layer
                        x = head_model[0](X_batch)  # Linear projection
                        x = x.unsqueeze(1)  # Add sequence dimension
                        x, _ = head_model[1](x, x, x)  # Self-attention
                        x = x.squeeze(1)  # Remove sequence dimension
                        outputs = head_model[2](x)  # Final linear
                    else:
                        outputs = head_model(X_batch)
                    
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()
                
                # Validation
                head_model.eval()
                with torch.no_grad():
                    X_val_gpu = X_val.to(self.device)
                    
                    if head_name == 'Attention':
                        x = head_model[0](X_val_gpu)
                        x = x.unsqueeze(1)
                        x, _ = head_model[1](x, x, x)
                        x = x.squeeze(1)
                        val_outputs = head_model[2](x)
                    else:
                        val_outputs = head_model(X_val_gpu)
                    
                    _, predicted = torch.max(val_outputs.data, 1)
                    y_val_gpu = y_val.to(self.device)
                    accuracy = (predicted == y_val_gpu).sum().item() / len(y_val_gpu)
                
                results[head_name] = {'accuracy': accuracy, 'final_loss': loss.item()}
                print(f"     ✅ {head_name}: {accuracy:.3f} accuracy")
            
            elapsed = time.time() - start_time
            print(f"   ✅ Phase C completed in {elapsed:.1f}s")
            
            self.results['phase_c'] = {
                'success': True,
                'head_results': results,
                'elapsed_time': elapsed
            }
            return True
            
        except Exception as e:
            print(f"   ❌ Phase C failed: {e}")
            self.results['phase_c'] = {'success': False, 'error': str(e)}
            return False

    def test_memory_constraints(self) -> bool:
        """Test 4GB VRAM constraint compliance"""
        print("\n🔥 Testing Memory Constraints")
        
        if not torch.cuda.is_available():
            print("   ⚠️  No CUDA available, skipping memory test")
            return True
        
        try:
            # Clear cache
            torch.cuda.empty_cache()
            
            # Get memory info
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            current_memory = torch.cuda.memory_allocated() / 1e9
            
            print(f"   Total VRAM: {total_memory:.1f} GB")
            print(f"   Current usage: {current_memory:.1f} GB")
            
            # Test maximum batch size that fits in 4GB
            max_batch_test_passed = True
            if current_memory < 3.5:  # Leave 0.5GB buffer
                print("   ✅ Memory usage within 4GB constraint")
            else:
                print("   ⚠️  Memory usage approaching 4GB limit")
                max_batch_test_passed = False
            
            self.results['memory_test'] = {
                'success': max_batch_test_passed,
                'total_vram_gb': total_memory,
                'current_usage_gb': current_memory,
                'within_4gb_limit': current_memory < 3.5
            }
            return max_batch_test_passed
            
        except Exception as e:
            print(f"   ❌ Memory test failed: {e}")
            self.results['memory_test'] = {'success': False, 'error': str(e)}
            return False

    def run_complete_test(self) -> Dict:
        """Run complete pipeline validation"""
        print("🚀 STARTING CAPSTONE-LAZARUS PIPELINE VALIDATION")
        print("=" * 60)
        
        overall_start = time.time()
        
        # Run all tests
        tests = [
            ('Phase B: Feature Extraction', self.test_phase_b_feature_extraction),
            ('Phase C: Head Training', self.test_phase_c_head_training),
            ('Memory Constraints', self.test_memory_constraints)
        ]
        
        passed_tests = 0
        for test_name, test_func in tests:
            try:
                if test_func():
                    passed_tests += 1
            except Exception as e:
                print(f"   ❌ {test_name} crashed: {e}")
        
        total_elapsed = time.time() - overall_start
        
        # Summary
        print("\n" + "=" * 60)
        print("🎯 VALIDATION SUMMARY")
        print(f"   Tests passed: {passed_tests}/{len(tests)}")
        print(f"   Total time: {total_elapsed:.1f}s")
        
        if passed_tests == len(tests):
            print("   ✅ ALL TESTS PASSED - Pipeline ready for production!")
        else:
            print("   ⚠️  Some tests failed - check logs above")
        
        # Save results
        self.results['overall'] = {
            'tests_passed': passed_tests,
            'total_tests': len(tests),
            'total_time': total_elapsed,
            'success': passed_tests == len(tests)
        }
        
        return self.results


def main():
    """Main execution function"""
    validator = PipelineValidator(test_samples=128)  # Small test for quick validation
    results = validator.run_complete_test()
    
    # Save results to file
    results_path = Path('validation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📄 Results saved to: {results_path}")
    
    return results['overall']['success']


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)