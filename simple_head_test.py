#!/usr/bin/env python3
"""
🔥 SIMPLE HEAD TRAINING TEST - No Multiprocessing

This script tests head training without DataLoader multiprocessing issues:
1. Loads features directly (no DataLoader)
2. Trains simple heads without multiprocessing
3. Validates the core training pipeline works
4. Provides baseline accuracy metrics

Target: HP ZBook Quadro P2000 (4GB VRAM)
"""

import sys
sys.path.append('src')

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import time
from datetime import datetime

class SimpleHeadTrainer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_dim = 1280  # EfficientNet-B0
        
        print(f"🔥 SIMPLE HEAD TRAINER")
        print(f"   Device: {self.device}")
        
    def load_all_features(self, manifest_path: str):
        """Load all features into memory"""
        print("📋 Loading features...")
        
        manifest = pd.read_csv(manifest_path)
        print(f"   Found {len(manifest)} feature files")
        
        features_list = []
        labels_list = []
        successful_loads = 0
        
        for idx, row in manifest.iterrows():
            try:
                # Load feature file
                data = np.load(row['feature_file'])
                feature = data['features'].astype(np.float32)
                
                features_list.append(feature)
                labels_list.append(row['class_name'])
                successful_loads += 1
                
                if successful_loads % 50 == 0:
                    print(f"   Loaded {successful_loads}/{len(manifest)} features")
                    
            except Exception as e:
                print(f"   ⚠️ Failed to load {row['feature_file']}: {e}")
                continue
        
        if not features_list:
            raise ValueError("No features loaded successfully")
        
        # Stack features
        X = np.stack(features_list)
        
        # Encode labels
        le = LabelEncoder()
        y = le.fit_transform(labels_list)
        
        print(f"✅ Loaded dataset:")
        print(f"   Features: {X.shape}")
        print(f"   Classes: {len(le.classes_)} {list(le.classes_)}")
        
        return X, y, le
    
    def create_heads(self, num_classes: int):
        """Create head architectures"""
        heads = {}
        
        # Linear head
        heads['linear'] = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, num_classes)
        )
        
        # MLP head  
        heads['mlp'] = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        # Simple attention head
        heads['attention'] = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        print(f"🏗️ Created {len(heads)} head architectures")
        return heads
    
    def train_head(self, head_model, X_train, y_train, X_val, y_val, lr=0.001, epochs=10):
        """Train a single head"""
        model = head_model.to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        # Convert to tensors
        X_train_tensor = torch.tensor(X_train).to(self.device)
        y_train_tensor = torch.tensor(y_train).to(self.device)
        X_val_tensor = torch.tensor(X_val).to(self.device)
        y_val_tensor = torch.tensor(y_val).to(self.device)
        
        best_val_acc = 0
        train_losses = []
        val_accuracies = []
        
        for epoch in range(epochs):
            # Training
            model.train()
            optimizer.zero_grad()
            
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                _, predicted = torch.max(val_outputs.data, 1)
                val_acc = (predicted == y_val_tensor).float().mean().item()
                val_accuracies.append(val_acc)
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
            
            if epoch % 2 == 0:
                print(f"      Epoch {epoch+1:2d}: Loss={loss.item():.4f}, Val_Acc={val_acc:.4f}")
        
        return {
            'best_val_accuracy': best_val_acc,
            'final_loss': train_losses[-1],
            'train_losses': train_losses,
            'val_accuracies': val_accuracies,
            'model': model
        }
    
    def run_experiments(self, manifest_path: str):
        """Run head training experiments"""
        print("🧪 Starting head training experiments...")
        
        # Load data
        X, y, label_encoder = self.load_all_features(manifest_path)
        num_classes = len(label_encoder.classes_)
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.2, stratify=y_temp, random_state=42
        )
        
        print(f"📊 Data splits:")
        print(f"   Train: {len(X_train)} samples")
        print(f"   Val: {len(X_val)} samples")
        print(f"   Test: {len(X_test)} samples")
        
        # Create heads
        heads = self.create_heads(num_classes)
        
        # Training parameters
        learning_rates = [0.001, 0.0003]
        results = {}
        
        for head_name, head_model in heads.items():
            print(f"\n🏗️ Training {head_name} head...")
            head_results = {}
            
            for lr in learning_rates:
                print(f"   📈 LR: {lr}")
                
                # Create fresh model copy
                import copy
                model_copy = copy.deepcopy(head_model)
                
                start_time = time.time()
                result = self.train_head(
                    model_copy, X_train, y_train, X_val, y_val, lr=lr, epochs=10
                )
                training_time = time.time() - start_time
                
                result['training_time'] = training_time
                result['learning_rate'] = lr
                
                head_results[f"lr_{lr}"] = result
                
                print(f"      ✅ Best Val Acc: {result['best_val_accuracy']:.4f} ({training_time:.1f}s)")
        
            results[head_name] = head_results
        
        # Find best model
        best_acc = 0
        best_config = None
        best_model = None
        
        for head_name, head_results in results.items():
            for lr_config, result in head_results.items():
                if result['best_val_accuracy'] > best_acc:
                    best_acc = result['best_val_accuracy']
                    best_config = f"{head_name}_{lr_config}"
                    best_model = result['model']
        
        print(f"\n🏆 BEST MODEL: {best_config}")
        print(f"   📊 Best Val Accuracy: {best_acc:.4f}")
        
        # Test on test set
        if best_model is not None:
            best_model.eval()
            X_test_tensor = torch.tensor(X_test).to(self.device)
            y_test_tensor = torch.tensor(y_test).to(self.device)
            
            with torch.no_grad():
                test_outputs = best_model(X_test_tensor)
                _, predicted = torch.max(test_outputs.data, 1)
                test_acc = (predicted == y_test_tensor).float().mean().item()
                
            print(f"   🧪 Test Accuracy: {test_acc:.4f}")
        
        return results


def main():
    """Main execution"""
    print("🔥 SIMPLE HEAD TRAINING TEST")
    print("=" * 50)
    
    trainer = SimpleHeadTrainer()
    
    # Use the manifest created by robust feature extraction
    manifest_path = "features/manifest_features.v001.csv"
    
    if not Path(manifest_path).exists():
        print(f"❌ Manifest not found: {manifest_path}")
        print("   Run robust_feature_extract.py first")
        return False
    
    try:
        results = trainer.run_experiments(manifest_path)
        
        print(f"\n✅ SUCCESS! Head training pipeline working")
        print(f"   🎯 All architectures trained successfully")  
        print(f"   📊 Ready for full ablation studies")
        
        return True
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)