#!/usr/bin/env python3
"""
🔥 SIMPLE HEAD TRAINING TEST (Fixed Paths)
Test head training with corrected Windows file path handling
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

class FeatureDataset(Dataset):
    """Dataset for pre-extracted features"""
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class LinearHead(nn.Module):
    """Simple linear classifier"""
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        x = self.dropout(x)
        return self.classifier(x)

class MLPHead(nn.Module):
    """Multi-layer perceptron classifier"""
    def __init__(self, input_dim, num_classes, hidden_dim=512):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(x)

class AttentionHead(nn.Module):
    """Attention-based classifier"""
    def __init__(self, input_dim, num_classes, hidden_dim=256):
        super().__init__()
        self.attention = nn.MultiheadAttention(input_dim, 8, batch_first=True)
        self.norm = nn.LayerNorm(input_dim)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        # Add sequence dimension for attention
        x = x.unsqueeze(1)  # [batch, 1, features]
        attn_output, _ = self.attention(x, x, x)
        x = self.norm(x + attn_output)
        x = x.squeeze(1)  # [batch, features]
        return self.classifier(x)

class SimpleHeadTrainer:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print(f"🔥 SIMPLE HEAD TRAINER")
        print(f"   Device: {self.device}")
    
    def map_actual_files(self, manifest_path):
        """Create mapping from manifest to actual files on disk"""
        # Read manifest
        df = pd.read_csv(manifest_path)
        
        # Get actual files in directory
        features_dir = "features/encoder_efficientnet_b0"
        actual_files = set(os.listdir(features_dir))
        
        # Map manifest entries to actual files
        file_mapping = {}
        for _, row in df.iterrows():
            manifest_file = os.path.basename(row['feature_file'])
            
            # Try exact match first
            if manifest_file in actual_files:
                file_mapping[manifest_file] = manifest_file
                continue
            
            # Find best match by checking truncated names
            best_match = None
            for actual_file in actual_files:
                if actual_file.startswith(manifest_file[:100]) and actual_file.endswith('.npz'):
                    best_match = actual_file
                    break
            
            if best_match:
                file_mapping[manifest_file] = best_match
            else:
                # Try matching by image ID
                img_id = manifest_file.split('_')[0] + '_' + manifest_file.split('_')[1]
                for actual_file in actual_files:
                    if actual_file.startswith(img_id) and actual_file.endswith('.npz'):
                        file_mapping[manifest_file] = actual_file
                        break
        
        return file_mapping, df
    
    def load_all_features(self, manifest_path):
        """Load all features with corrected file paths"""
        print("📋 Loading features...")
        
        file_mapping, df = self.map_actual_files(manifest_path)
        features_dir = "features/encoder_efficientnet_b0"
        
        features_list = []
        labels_list = []
        
        loaded_count = 0
        for _, row in df.iterrows():
            manifest_file = os.path.basename(row['feature_file'])
            
            if manifest_file not in file_mapping:
                continue
                
            actual_file = file_mapping[manifest_file]
            file_path = os.path.join(features_dir, actual_file)
            
            try:
                data = np.load(file_path)
                features = data['features']
                features_list.append(features)
                labels_list.append(row['class_name'])
                loaded_count += 1
                
                if loaded_count % 50 == 0:
                    print(f"   Loaded {loaded_count} features...")
                    
            except Exception as e:
                continue
        
        print(f"   Successfully loaded {loaded_count} features")
        
        if loaded_count == 0:
            raise ValueError("No features loaded successfully")
        
        # Convert to arrays
        X = np.array(features_list)
        y = np.array(labels_list)
        
        # Encode labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        print(f"   Feature shape: {X.shape}")
        print(f"   Classes: {len(label_encoder.classes_)}")
        
        return X, y_encoded, label_encoder
    
    def create_head(self, head_type, input_dim, num_classes):
        """Create classification head"""
        if head_type == 'linear':
            return LinearHead(input_dim, num_classes)
        elif head_type == 'mlp':
            return MLPHead(input_dim, num_classes)
        elif head_type == 'attention':
            return AttentionHead(input_dim, num_classes)
        else:
            raise ValueError(f"Unknown head type: {head_type}")
    
    def train_head(self, model, train_loader, val_loader, lr=0.001, epochs=20):
        """Train classification head"""
        model = model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        best_val_acc = 0.0
        best_model_state = None
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for features, labels in train_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for features, labels in val_loader:
                    features, labels = features.to(self.device), labels.to(self.device)
                    outputs = model(features)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            train_acc = 100 * train_correct / train_total
            val_acc = 100 * val_correct / val_total
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()
            
            if (epoch + 1) % 5 == 0:
                print(f"      Epoch {epoch+1:2d}: Train Acc: {train_acc:.1f}%, Val Acc: {val_acc:.1f}%")
        
        # Load best model
        if best_model_state:
            model.load_state_dict(best_model_state)
        
        return model, best_val_acc
    
    def run_experiments(self, manifest_path):
        """Run head training experiments"""
        print("🧪 Starting head training experiments...")
        
        # Load features
        X, y, label_encoder = self.load_all_features(manifest_path)
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.67, random_state=42, stratify=y_temp  # 0.67 of 0.3 = 0.2 total
        )
        
        print(f"📊 Data splits: Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Create datasets (NO DataLoader multiprocessing)
        train_dataset = FeatureDataset(X_train, y_train)
        val_dataset = FeatureDataset(X_val, y_val)
        test_dataset = FeatureDataset(X_test, y_test)
        
        # Create data loaders with num_workers=0
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
        
        input_dim = X.shape[1]
        num_classes = len(np.unique(y))
        
        # Experiment configurations
        experiments = [
            ('linear', [0.01, 0.001, 0.0001]),
            ('mlp', [0.01, 0.001, 0.0001]),
            ('attention', [0.001, 0.0001, 0.00001])
        ]
        
        results = []
        
        for head_type, learning_rates in experiments:
            print(f"\n🎯 Testing {head_type.upper()} head:")
            
            for lr in learning_rates:
                print(f"   📈 Learning rate: {lr}")
                
                # Create and train model
                model = self.create_head(head_type, input_dim, num_classes)
                trained_model, val_acc = self.train_head(
                    model, train_loader, val_loader, lr=lr, epochs=15
                )
                
                # Test performance
                trained_model.eval()
                test_correct = 0
                test_total = 0
                
                with torch.no_grad():
                    for features, labels in test_loader:
                        features, labels = features.to(self.device), labels.to(self.device)
                        outputs = trained_model(features)
                        _, predicted = torch.max(outputs.data, 1)
                        test_total += labels.size(0)
                        test_correct += (predicted == labels).sum().item()
                
                test_acc = 100 * test_correct / test_total
                
                result = {
                    'head_type': head_type,
                    'learning_rate': lr,
                    'val_accuracy': val_acc,
                    'test_accuracy': test_acc
                }
                results.append(result)
                
                print(f"      ✅ Best Val Acc: {val_acc:.1f}%, Test Acc: {test_acc:.1f}%")
        
        return results

def main():
    print("🔥 SIMPLE HEAD TRAINING TEST (Fixed)")
    print("="*50)
    
    # Initialize trainer
    trainer = SimpleHeadTrainer()
    
    manifest_path = "features/manifest_features.v001.csv"
    
    try:
        # Run experiments
        results = trainer.run_experiments(manifest_path)
        
        print("\n🏆 EXPERIMENT RESULTS:")
        print("="*50)
        
        # Sort by test accuracy
        results.sort(key=lambda x: x['test_accuracy'], reverse=True)
        
        for i, result in enumerate(results):
            print(f"{i+1:2d}. {result['head_type'].upper():9} (lr={result['learning_rate']:7}) - "
                  f"Val: {result['val_accuracy']:5.1f}%, Test: {result['test_accuracy']:5.1f}%")
        
        # Best result
        best = results[0]
        print(f"\n🥇 BEST: {best['head_type'].upper()} with lr={best['learning_rate']}")
        print(f"   Test Accuracy: {best['test_accuracy']:.1f}%")
        
        print("\n✅ HEAD TRAINING TEST COMPLETE!")
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())