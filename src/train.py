#!/usr/bin/env python3
"""
CLI trainer for balanced subset training
Mirrors the Jupyter notebook functionality for production use
"""
import argparse
import os
import sys
import time
from pathlib import Path
import json
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import timm
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent))
from data_utils import ImageFolderAlb, get_transforms

def setup_logging(log_file=None):
    """Setup logging configuration"""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    if log_file:
        logging.basicConfig(level=logging.INFO, format=log_format, 
                          handlers=[
                              logging.FileHandler(log_file),
                              logging.StreamHandler()
                          ])
    else:
        logging.basicConfig(level=logging.INFO, format=log_format)
    
    return logging.getLogger(__name__)

class SubsetTrainer:
    """Trainer for balanced subset experiments"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = setup_logging(args.log_file)
        
        # Setup model save directory
        self.save_dir = Path(args.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"🚀 SubsetTrainer initialized")
        self.logger.info(f"   Device: {self.device}")
        self.logger.info(f"   Save directory: {self.save_dir}")
        
    def setup_data(self):
        """Setup data loaders"""
        self.logger.info("📁 Setting up data loaders...")
        
        # Verify subset directory exists
        subset_dir = Path(self.args.subset_dir)
        if not subset_dir.exists():
            raise FileNotFoundError(f"Subset directory not found: {subset_dir}")
        
        train_dir = subset_dir / "train"
        val_dir = subset_dir / "val"
        
        if not train_dir.exists() or not val_dir.exists():
            raise FileNotFoundError("Train/val directories not found in subset")
        
        # Get transforms
        train_transform, val_transform = get_transforms(
            img_size=self.args.img_size,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # Create datasets
        train_dataset = ImageFolderAlb(train_dir, transform=train_transform)
        val_dataset = ImageFolderAlb(val_dir, transform=val_transform)
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        self.num_classes = len(train_dataset.classes)
        self.class_names = train_dataset.classes
        
        self.logger.info(f"   Classes: {self.num_classes}")
        self.logger.info(f"   Train samples: {len(train_dataset)}")
        self.logger.info(f"   Val samples: {len(val_dataset)}")
        
    def setup_model(self):
        """Setup model with frozen backbone"""
        self.logger.info("🤖 Setting up model...")
        
        # Create model
        self.model = timm.create_model(
            'efficientnet_b0',
            pretrained=True,
            num_classes=self.num_classes
        )
        
        # Freeze backbone for transfer learning
        for name, param in self.model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False
        
        self.model = self.model.to(self.device)
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        self.logger.info(f"   Model: EfficientNet-B0")
        self.logger.info(f"   Trainable params: {trainable_params:,}")
        self.logger.info(f"   Total params: {total_params:,}")
        self.logger.info(f"   Frozen backbone: True")
        
    def setup_training(self):
        """Setup loss, optimizer, and training utilities"""
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay
        )
        self.scaler = GradScaler() if torch.cuda.is_available() else None
        
        self.logger.info("⚙️ Training setup:")
        self.logger.info(f"   Learning rate: {self.args.learning_rate}")
        self.logger.info(f"   Weight decay: {self.args.weight_decay}")
        self.logger.info(f"   AMP enabled: {self.scaler is not None}")
        
    def train_one_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        losses, preds, targets = [], [], []
        
        loop = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.epochs}")
        for batch_idx, (imgs, labels) in enumerate(loop):
            imgs, labels = imgs.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            
            # Forward pass with AMP
            with autocast(enabled=(self.scaler is not None)):
                outputs = self.model(imgs)
                loss = self.criterion(outputs, labels)
            
            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            # Track metrics
            losses.append(loss.item())
            preds.extend(outputs.argmax(1).cpu().numpy())
            targets.extend(labels.cpu().numpy())
            
            # Update progress bar
            if len(losses) > 0:
                current_loss = np.mean(losses)
                current_acc = accuracy_score(targets, preds)
                loop.set_postfix({
                    'loss': f'{current_loss:.4f}',
                    'acc': f'{current_acc:.4f}'
                })
        
        return np.mean(losses), accuracy_score(targets, preds)
    
    @torch.no_grad()
    def validate(self):
        """Validate model performance"""
        self.model.eval()
        losses, preds, targets = [], [], []
        
        for imgs, labels in tqdm(self.val_loader, desc="Validating", leave=False):
            imgs, labels = imgs.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
            
            outputs = self.model(imgs)
            loss = self.criterion(outputs, labels)
            
            losses.append(loss.item())
            preds.extend(outputs.argmax(1).cpu().numpy())
            targets.extend(labels.cpu().numpy())
        
        return np.mean(losses), accuracy_score(targets, preds)
    
    def save_checkpoint(self, epoch, is_best, metrics):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'args': vars(self.args)
        }
        
        # Save latest
        latest_path = self.save_dir / 'latest.pth'
        torch.save(checkpoint, latest_path)
        
        # Save best
        if is_best:
            best_path = self.save_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            self.logger.info(f"💾 Saved best model! (Val Acc: {metrics['val_acc']:.4f})")
    
    def train(self):
        """Main training loop"""
        self.logger.info("🎯 Starting training...")
        
        best_acc = 0.0
        training_history = {
            'train_loss': [], 'train_acc': [], 
            'val_loss': [], 'val_acc': []
        }
        start_time = time.time()
        
        for epoch in range(self.args.epochs):
            self.logger.info(f"Epoch {epoch+1}/{self.args.epochs}")
            
            # Training
            train_loss, train_acc = self.train_one_epoch(epoch)
            
            # Validation
            val_loss, val_acc = self.validate()
            
            # Save metrics
            metrics = {
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc
            }
            
            training_history['train_loss'].append(train_loss)
            training_history['train_acc'].append(train_acc)
            training_history['val_loss'].append(val_loss)
            training_history['val_acc'].append(val_acc)
            
            # Log results
            self.logger.info(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            self.logger.info(f"Val Loss:   {val_loss:.4f}   | Val Acc:   {val_acc:.4f}")
            
            # Save checkpoint
            is_best = val_acc > best_acc
            if is_best:
                best_acc = val_acc
            
            self.save_checkpoint(epoch, is_best, metrics)
        
        # Training completed
        elapsed_time = time.time() - start_time
        self.logger.info("🏁 Training completed!")
        self.logger.info(f"⏱️ Total time: {elapsed_time:.1f}s ({elapsed_time/60:.1f}m)")
        self.logger.info(f"🎯 Best validation accuracy: {best_acc:.4f}")
        
        # Save training history
        history_path = self.save_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)
        
        return best_acc, training_history

def main():
    parser = argparse.ArgumentParser(
        description="CLI trainer for balanced subset training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Path to original dataset directory')
    parser.add_argument('--subset-dir', type=str, required=True,
                       help='Path to subset directory (created by create_subset.py)')
    
    # Subset parameters
    parser.add_argument('--samples-per-class', type=int, default=50,
                       help='Number of samples per class in subset')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=4,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay')
    
    # Model parameters
    parser.add_argument('--img-size', type=int, default=160,
                       help='Input image size')
    
    # Output parameters
    parser.add_argument('--save-dir', type=str, default='./subset_training_results',
                       help='Directory to save results')
    parser.add_argument('--log-file', type=str, default=None,
                       help='Path to log file (optional)')
    
    # Other options
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Create trainer and run training
    trainer = SubsetTrainer(args)
    trainer.setup_data()
    trainer.setup_model()
    trainer.setup_training()
    
    best_acc, history = trainer.train()
    
    print(f"\n✅ Training completed successfully!")
    print(f"🎯 Best accuracy: {best_acc:.4f}")
    print(f"📁 Results saved to: {trainer.save_dir}")

if __name__ == "__main__":
    main()