"""
CAPSTONE-LAZARUS: PyTorch Training Pipeline
==========================================
Production-ready training with AMP, checkpointing, and HP ZBook G5 optimizations.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau

import numpy as np
import pandas as pd
from pathlib import Path
import logging
import time
import json
from typing import Dict, Any, Optional, Tuple, List
from collections import defaultdict
from tqdm.auto import tqdm

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Local imports
from .model_factory_torch import create_model
from .data_utils_torch import make_dataloaders, create_subset_loader

logger = logging.getLogger(__name__)

# Weights & Biases integration (optional)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("Weights & Biases not available. Install with: pip install wandb")

# ONNX export support
try:
    import onnx
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logger.warning("ONNX not available. Install with: pip install onnx")


class EMA:
    """Exponential Moving Average for model parameters."""
    
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update shadow parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """Replace model parameters with shadow parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original model parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class MetricsTracker:
    """Track and compute training metrics."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.predictions = []
        self.targets = []
        self.losses = []
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, loss: float):
        """Update metrics with batch results."""
        # Convert to CPU and detach
        preds = predictions.detach().cpu().numpy()
        targs = targets.detach().cpu().numpy()
        
        if preds.ndim > 1 and preds.shape[1] > 1:
            # Multi-class: get argmax
            preds = np.argmax(preds, axis=1)
        
        self.predictions.extend(preds.tolist())
        self.targets.extend(targs.tolist())
        self.losses.append(loss)
    
    def compute(self) -> Dict[str, float]:
        """Compute final metrics."""
        if not self.predictions:
            return {}
        
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        # Basic metrics
        accuracy = accuracy_score(targets, predictions)
        avg_loss = np.mean(self.losses)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            targets, predictions, average=None, zero_division=0
        )
        
        # Overall weighted metrics
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            targets, predictions, average='weighted', zero_division=0
        )
        
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            'precision_per_class': precision.tolist(),
            'recall_per_class': recall.tolist(),
            'f1_per_class': f1.tolist(),
            'support_per_class': support.tolist()
        }
        
        return metrics


class Trainer:
    """Main training class with all production features."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        save_dir: str = "./checkpoints"
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        logger.info(f"Training on device: {self.device}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name()}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Mixed precision setup
        self.use_amp = config.get('use_amp', True) and torch.cuda.is_available()
        if self.use_amp:
            self.scaler = GradScaler()
            logger.info("Using Automatic Mixed Precision (AMP)")
        
        # Optimizer setup
        self.optimizer = self._create_optimizer()
        
        # Scheduler setup
        self.scheduler = self._create_scheduler()
        
        # EMA setup
        if config.get('use_ema', True):
            self.ema = EMA(model, decay=config.get('ema_decay', 0.9999))
        else:
            self.ema = None
        
        # Loss function
        self.criterion = self._create_loss_function()
        
        # Gradient accumulation
        self.accumulation_steps = config.get('accumulation_steps', 1)
        
        # Metrics tracking
        self.train_metrics = MetricsTracker()
        self.val_metrics = MetricsTracker()
        
        # History tracking
        self.history = defaultdict(list)
        
        # Best model tracking
        self.best_val_score = float('-inf')
        self.epochs_without_improvement = 0
        
        # Weights & Biases
        self.use_wandb = config.get('use_wandb', False) and WANDB_AVAILABLE
        if self.use_wandb:
            self._init_wandb()
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on config."""
        
        optimizer_name = self.config.get('optimizer', 'adamw').lower()
        lr = self.config.get('learning_rate', 1e-3)
        weight_decay = self.config.get('weight_decay', 1e-4)
        
        if optimizer_name == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=self.config.get('betas', (0.9, 0.999))
            )
        elif optimizer_name == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'sgd':
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=self.config.get('momentum', 0.9),
                weight_decay=weight_decay,
                nesterov=self.config.get('nesterov', True)
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        logger.info(f"Created {optimizer_name} optimizer with lr={lr}")
        return optimizer
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler."""
        
        scheduler_name = self.config.get('scheduler', 'onecycle').lower()
        
        if scheduler_name == 'none':
            return None
        
        max_lr = self.config.get('learning_rate', 1e-3)
        epochs = self.config.get('epochs', 50)
        steps_per_epoch = len(self.train_loader) // self.accumulation_steps
        
        if scheduler_name == 'onecycle':
            scheduler = OneCycleLR(
                self.optimizer,
                max_lr=max_lr,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=self.config.get('onecycle_pct_start', 0.3),
                anneal_strategy='cos'
            )
        elif scheduler_name == 'cosine':
            scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.config.get('cosine_t0', 10),
                T_mult=self.config.get('cosine_tmult', 2),
                eta_min=self.config.get('cosine_min_lr', 1e-6)
            )
        elif scheduler_name == 'reduce_on_plateau':
            scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=self.config.get('reduce_factor', 0.5),
                patience=self.config.get('reduce_patience', 5),
                verbose=True
            )
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")
        
        logger.info(f"Created {scheduler_name} scheduler")
        return scheduler
    
    def _create_loss_function(self) -> nn.Module:
        """Create loss function with optional class weighting."""
        
        loss_name = self.config.get('loss_function', 'crossentropy').lower()
        
        if loss_name == 'crossentropy':
            # Use class weights if available
            class_weights = self.config.get('class_weights')
            if class_weights:
                class_weights = torch.tensor(class_weights, dtype=torch.float32, device=self.device)
            
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        elif loss_name == 'focal':
            # Implement focal loss for imbalanced datasets
            alpha = self.config.get('focal_alpha', 1.0)
            gamma = self.config.get('focal_gamma', 2.0)
            criterion = FocalLoss(alpha=alpha, gamma=gamma)
        else:
            raise ValueError(f"Unknown loss function: {loss_name}")
        
        logger.info(f"Created {loss_name} loss function")
        return criterion
    
    def _init_wandb(self):
        """Initialize Weights & Biases logging."""
        
        wandb.init(
            project=self.config.get('wandb_project', 'plant-disease-classification'),
            name=self.config.get('wandb_run_name', 'pytorch-training'),
            config=self.config,
            tags=self.config.get('wandb_tags', ['pytorch', 'plant-disease'])
        )
        
        # Watch model
        wandb.watch(self.model, log='all', log_freq=100)
        logger.info("Initialized Weights & Biases logging")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        
        self.model.train()
        self.train_metrics.reset()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]")
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass with optional mixed precision
            if self.use_amp:
                with autocast():
                    output = self.model(data)
                    loss = self.criterion(output, target) / self.accumulation_steps
                
                # Backward pass
                self.scaler.scale(loss).backward()
            else:
                output = self.model(data)
                loss = self.criterion(output, target) / self.accumulation_steps
                loss.backward()
            
            # Update metrics
            self.train_metrics.update(output, target, loss.item() * self.accumulation_steps)
            
            # Optimizer step with gradient accumulation
            if (batch_idx + 1) % self.accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                # Update EMA
                if self.ema:
                    self.ema.update()
                
                # Update scheduler (if step-based)
                if self.scheduler and not isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{loss.item() * self.accumulation_steps:.4f}",
                'LR': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
        
        return self.train_metrics.compute()
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Validate for one epoch."""
        
        self.model.eval()
        self.val_metrics.reset()
        
        # Use EMA parameters for validation if available
        if self.ema:
            self.ema.apply_shadow()
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1} [Val]")
            
            for data, target in pbar:
                data, target = data.to(self.device), target.to(self.device)
                
                if self.use_amp:
                    with autocast():
                        output = self.model(data)
                        loss = self.criterion(output, target)
                else:
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                # Update metrics
                self.val_metrics.update(output, target, loss.item())
                
                # Update progress bar
                pbar.set_postfix({'Loss': f"{loss.item():.4f}"})
        
        # Restore original parameters
        if self.ema:
            self.ema.restore()
        
        return self.val_metrics.compute()
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint."""
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.use_amp else None,
            'ema_state_dict': self.ema.shadow if self.ema else None,
            'metrics': metrics,
            'config': self.config,
            'best_val_score': self.best_val_score
        }
        
        # Save latest checkpoint
        checkpoint_path = self.save_dir / "checkpoint_latest.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.save_dir / "checkpoint_best.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model with validation score: {metrics.get('accuracy', 0):.4f}")
        
        # Save periodic checkpoint
        if (epoch + 1) % self.config.get('save_every', 10) == 0:
            periodic_path = self.save_dir / f"checkpoint_epoch_{epoch+1:03d}.pth"
            torch.save(checkpoint, periodic_path)
    
    def load_checkpoint(self, checkpoint_path: str, resume_training: bool = True):
        """Load model checkpoint."""
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if resume_training:
            # Load optimizer state
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load scheduler state
            if self.scheduler and checkpoint.get('scheduler_state_dict'):
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # Load scaler state
            if self.use_amp and checkpoint.get('scaler_state_dict'):
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            # Load EMA state
            if self.ema and checkpoint.get('ema_state_dict'):
                self.ema.shadow = checkpoint['ema_state_dict']
            
            # Load training state
            self.best_val_score = checkpoint.get('best_val_score', float('-inf'))
            
            logger.info(f"Resumed training from epoch {checkpoint['epoch'] + 1}")
            return checkpoint['epoch'] + 1
        else:
            logger.info("Loaded model weights for inference")
            return 0
    
    def train(self, epochs: Optional[int] = None, resume_from: Optional[str] = None) -> Dict[str, List]:
        """Main training loop."""
        
        if epochs is None:
            epochs = self.config.get('epochs', 50)
        
        start_epoch = 0
        if resume_from:
            start_epoch = self.load_checkpoint(resume_from, resume_training=True)
        
        logger.info(f"Starting training for {epochs - start_epoch} epochs...")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        
        for epoch in range(start_epoch, epochs):
            epoch_start_time = time.time()
            
            # Training phase
            train_metrics = self.train_epoch(epoch)
            
            # Validation phase
            val_metrics = self.validate_epoch(epoch)
            
            # Update scheduler (if epoch-based)
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_metrics['accuracy'])
            
            # Track metrics
            for key, value in train_metrics.items():
                self.history[f'train_{key}'].append(value)
            for key, value in val_metrics.items():
                self.history[f'val_{key}'].append(value)
            
            # Check for improvement
            val_score = val_metrics.get('accuracy', 0)
            is_best = val_score > self.best_val_score
            if is_best:
                self.best_val_score = val_score
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_metrics, is_best)
            
            # Logging
            epoch_time = time.time() - epoch_start_time
            logger.info(
                f"Epoch {epoch+1:3d}/{epochs}: "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train Acc: {train_metrics['accuracy']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.4f}, "
                f"Time: {epoch_time:.1f}s"
            )
            
            # Weights & Biases logging
            if self.use_wandb:
                log_dict = {
                    'epoch': epoch + 1,
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                    'epoch_time': epoch_time
                }
                log_dict.update({f'train_{k}': v for k, v in train_metrics.items()})
                log_dict.update({f'val_{k}': v for k, v in val_metrics.items()})
                wandb.log(log_dict)
            
            # Early stopping
            patience = self.config.get('early_stopping_patience', 15)
            if self.epochs_without_improvement >= patience:
                logger.info(f"Early stopping after {epoch+1} epochs (patience: {patience})")
                break
        
        logger.info(f"Training completed! Best validation accuracy: {self.best_val_score:.4f}")
        
        if self.use_wandb:
            wandb.finish()
        
        return dict(self.history)
    
    def export_onnx(self, output_path: str, input_size: Tuple[int, ...] = (1, 3, 224, 224)):
        """Export model to ONNX format."""
        
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX not available. Install with: pip install onnx")
        
        self.model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(*input_size, device=self.device)
        
        # Export
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        logger.info(f"Model exported to ONNX: {output_path}")
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Loss Over Time')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy plot
        axes[0, 1].plot(self.history['train_accuracy'], label='Train Accuracy')
        axes[0, 1].plot(self.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 1].set_title('Accuracy Over Time')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # F1 Score plot
        axes[1, 0].plot(self.history['train_f1_weighted'], label='Train F1')
        axes[1, 0].plot(self.history['val_f1_weighted'], label='Validation F1')
        axes[1, 0].set_title('F1 Score Over Time')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning rate plot (if available)
        if hasattr(self, 'lr_history'):
            axes[1, 1].plot(self.lr_history)
            axes[1, 1].set_title('Learning Rate Schedule')
            axes[1, 1].set_xlabel('Step')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].grid(True)
        else:
            axes[1, 1].text(0.5, 0.5, 'LR History Not Available', 
                          ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to {save_path}")
        else:
            plt.show()


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce_loss(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss


def main():
    """Example training script."""
    
    # This would be moved to a separate training script
    import yaml
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create model
    model = create_model(config)
    
    # Create data loaders
    train_loader, val_loader = make_dataloaders(
        data_dir=config['data_dir'],
        config=config
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        save_dir=config.get('save_dir', './checkpoints')
    )
    
    # Train model
    history = trainer.train()
    
    # Plot results
    trainer.plot_training_history(save_path='training_history.png')
    
    # Export to ONNX
    trainer.export_onnx('model.onnx')
    
    print("Training completed successfully!")


if __name__ == "__main__":
    main()