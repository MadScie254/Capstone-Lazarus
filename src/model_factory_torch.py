"""
CAPSTONE-LAZARUS: PyTorch Model Factory
======================================
Fast, efficient model architectures using timm with optional quantum layers.
Optimized for HP ZBook G5 and Colab scaling.
"""

import torch
import torch.nn as nn
import timm
from typing import Optional, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

# Backbone mapping for timm models
BACKBONE_MAP = {
    "tf_efficientnet_b0": "tf_efficientnet_b0.ns_jft_in1k",
    "efficientnet_b0": "efficientnet_b0.ra_in1k", 
    "resnet18": "resnet18.a1_in1k",
    "resnet50": "resnet50.a1_in1k",
    "mobilenetv3_small": "mobilenetv3_small_100.lamb_in1k",
    "mobilenetv3_large": "mobilenetv3_large_100.ra_in1k",
    "regnet_x_400mf": "regnet_x_400mf.pycls_in1k",
    "convnext_tiny": "convnext_tiny.fb_in1k",
}

class PlantDiseaseModel(nn.Module):
    """
    Plant Disease Classification Model with optional quantum layers.
    Optimized for transfer learning and fast inference.
    """
    
    def __init__(
        self,
        backbone: str = "tf_efficientnet_b0",
        num_classes: int = 19,
        pretrained: bool = True,
        dropout_rate: float = 0.3,
        use_quantum: bool = False,
        quantum_config: Optional[Dict] = None
    ):
        super().__init__()
        
        self.backbone_name = backbone
        self.num_classes = num_classes
        self.use_quantum = use_quantum
        
        # Load pretrained backbone
        if backbone not in BACKBONE_MAP:
            raise ValueError(f"Backbone '{backbone}' not supported. Choose from {list(BACKBONE_MAP.keys())}")
            
        model_name = BACKBONE_MAP[backbone]
        self.backbone = timm.create_model(
            model_name, 
            pretrained=pretrained,
            num_classes=0,  # Remove classifier head
            global_pool='',  # Remove global pooling
        )
        
        # Get feature dimensions
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            self.feature_dim = features.shape[1]
            
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classification head
        if use_quantum and quantum_config:
            self._build_quantum_head(quantum_config, dropout_rate)
        else:
            self._build_standard_head(dropout_rate)
            
        self._initialize_weights()
        
    def _build_standard_head(self, dropout_rate: float):
        """Build standard classification head."""
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(256, self.num_classes)
        )
        
    def _build_quantum_head(self, quantum_config: Dict, dropout_rate: float):
        """Build quantum-classical hybrid head."""
        try:
            from .quantum_layer import QuantumLayer
            
            embedding_dim = quantum_config.get('embedding_dim', 64)
            
            self.pre_quantum = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(self.feature_dim, embedding_dim),
                nn.BatchNorm1d(embedding_dim),
                nn.ReLU(inplace=True)
            )
            
            self.quantum_layer = QuantumLayer(
                n_qubits=quantum_config.get('n_qubits', 4),
                n_layers=quantum_config.get('n_layers', 3),
                embedding_dim=embedding_dim
            )
            
            self.post_quantum = nn.Sequential(
                nn.Dropout(dropout_rate / 2),
                nn.Linear(quantum_config.get('n_qubits', 4), 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Linear(128, self.num_classes)
            )
            
            logger.info(f"Quantum layer initialized with {quantum_config['n_qubits']} qubits")
            
        except ImportError:
            logger.warning("PennyLane not available, falling back to standard head")
            self.use_quantum = False
            self._build_standard_head(dropout_rate)
    
    def _initialize_weights(self):
        """Initialize weights for newly added layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract features
        features = self.backbone(x)
        
        # Global pooling
        features = self.global_pool(features)
        features = torch.flatten(features, 1)
        
        # Classification
        if self.use_quantum:
            x = self.pre_quantum(features)
            x = self.quantum_layer(x)
            x = self.post_quantum(x)
        else:
            x = self.classifier(features)
            
        return x
    
    def freeze_backbone(self):
        """Freeze backbone for transfer learning."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("Backbone frozen for transfer learning")
        
    def unfreeze_backbone(self):
        """Unfreeze backbone for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        logger.info("Backbone unfrozen for fine-tuning")
        
    def unfreeze_last_n_layers(self, n: int = 10):
        """Unfreeze last n layers for gradual fine-tuning."""
        backbone_layers = list(self.backbone.modules())
        for layer in backbone_layers[-n:]:
            for param in layer.parameters():
                param.requires_grad = True
        logger.info(f"Last {n} backbone layers unfrozen for fine-tuning")


def get_model(
    backbone: str = "tf_efficientnet_b0",
    num_classes: int = 19,
    pretrained: bool = True,
    use_quantum: bool = False,
    quantum_config: Optional[Dict] = None,
    **kwargs
) -> PlantDiseaseModel:
    """
    Factory function to create plant disease classification models.
    
    Args:
        backbone: Model architecture name
        num_classes: Number of plant disease classes
        pretrained: Use pretrained weights
        use_quantum: Enable quantum layers (experimental)
        quantum_config: Configuration for quantum layers
        **kwargs: Additional model parameters
        
    Returns:
        Initialized PlantDiseaseModel
    """
    
    model = PlantDiseaseModel(
        backbone=backbone,
        num_classes=num_classes,
        pretrained=pretrained,
        use_quantum=use_quantum,
        quantum_config=quantum_config,
        **kwargs
    )
    
    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Model created: {backbone}")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Quantum enabled: {use_quantum}")
    
    return model


def get_model_size_mb(model: nn.Module) -> float:
    """Calculate model size in MB."""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb


def print_model_summary(model: nn.Module, input_size: Tuple[int, ...] = (3, 224, 224)):
    """Print detailed model summary."""
    try:
        from torchsummary import summary
        summary(model, input_size)
    except ImportError:
        logger.warning("torchsummary not available, install with: pip install torchsummary")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        size_mb = get_model_size_mb(model)
        
        print(f"Model Summary:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Model size: {size_mb:.2f} MB")