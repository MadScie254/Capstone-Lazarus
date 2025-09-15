"""
Default Configuration for CAPSTONE-LAZARUS
==========================================
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import os
from pathlib import Path

@dataclass
class DataConfig:
    """Data processing configuration"""
    batch_size: int = 32
    validation_split: float = 0.2
    test_split: float = 0.1
    image_size: tuple = (224, 224)
    augmentation: bool = True
    normalize: bool = True
    cache: bool = True
    prefetch_buffer: int = 1000
    shuffle_buffer: int = 10000

@dataclass
class ModelConfig:
    """Model architecture configuration"""
    architecture: str = "efficient_net"
    num_classes: int = 10
    input_shape: tuple = (224, 224, 3)
    dropout_rate: float = 0.2
    l2_regularization: float = 1e-4
    use_mixed_precision: bool = True
    enable_pruning: bool = False
    enable_quantization: bool = False

@dataclass
class TrainingConfig:
    """Training configuration"""
    epochs: int = 100
    learning_rate: float = 1e-3
    optimizer: str = "adam"
    loss_function: str = "categorical_crossentropy"
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "top_5_accuracy"])
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5
    save_best_only: bool = True
    checkpoint_frequency: int = 5
    use_mixed_precision: bool = True
    gradient_clip_norm: Optional[float] = None

@dataclass
class NASConfig:
    """Neural Architecture Search configuration"""
    enabled: bool = False
    search_space: str = "mobile_net"
    max_epochs: int = 50
    num_trials: int = 100
    objective_metric: str = "val_accuracy"
    pruning_enabled: bool = True

@dataclass
class FederatedConfig:
    """Federated learning configuration"""
    enabled: bool = False
    num_rounds: int = 10
    num_clients: int = 5
    fraction_fit: float = 1.0
    fraction_eval: float = 1.0
    min_fit_clients: int = 2
    min_eval_clients: int = 2
    min_available_clients: int = 2

@dataclass
class DifferentialPrivacyConfig:
    """Differential Privacy configuration"""
    enabled: bool = False
    l2_norm_clip: float = 1.0
    noise_multiplier: float = 1.1
    num_microbatches: int = 1
    learning_rate: float = 0.15
    delta: float = 1e-5

@dataclass
class MonitoringConfig:
    """Monitoring and observability configuration"""
    drift_detection: bool = True
    drift_threshold: float = 0.05
    prometheus_enabled: bool = False
    prometheus_port: int = 8000
    log_level: str = "INFO"
    metrics_collection: bool = True

@dataclass
class MLflowConfig:
    """MLflow experiment tracking configuration"""
    enabled: bool = True
    tracking_uri: str = "file:./experiments/mlruns"
    experiment_name: str = "capstone_lazarus"
    artifact_location: str = "./experiments/artifacts"
    log_models: bool = True
    log_artifacts: bool = True

@dataclass
class WandbConfig:
    """Weights & Biases configuration"""
    enabled: bool = False
    project: str = "capstone-lazarus"
    entity: Optional[str] = None
    api_key_env: str = "WANDB_API_KEY"

@dataclass
class ExplainabilityConfig:
    """Model explainability configuration"""
    shap_enabled: bool = True
    integrated_gradients: bool = True
    counterfactuals: bool = True
    feature_importance: bool = True
    max_samples_explain: int = 100

@dataclass
class SecurityConfig:
    """Security and privacy configuration"""
    use_vault: bool = False
    vault_url: Optional[str] = None
    encrypt_artifacts: bool = False
    audit_logging: bool = True

@dataclass
class DockerConfig:
    """Docker deployment configuration"""
    base_image: str = "tensorflow/tensorflow:2.20.0"
    gpu_image: str = "tensorflow/tensorflow:2.20.0-gpu"
    port: int = 8501
    workers: int = 1
    memory_limit: str = "4G"
    cpu_limit: str = "2"

@dataclass
class Config:
    """Main configuration class"""
    # Basic settings
    project_name: str = "capstone-lazarus"
    task: str = "classification"  # classification, regression
    random_seed: int = 42
    device: str = "auto"  # auto, cpu, gpu
    
    # Component configurations
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    nas: NASConfig = field(default_factory=NASConfig)
    federated: FederatedConfig = field(default_factory=FederatedConfig)
    differential_privacy: DifferentialPrivacyConfig = field(default_factory=DifferentialPrivacyConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    explainability: ExplainabilityConfig = field(default_factory=ExplainabilityConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    docker: DockerConfig = field(default_factory=DockerConfig)
    
    # Paths
    data_dir: Path = field(default_factory=lambda: Path("data"))
    models_dir: Path = field(default_factory=lambda: Path("models"))
    experiments_dir: Path = field(default_factory=lambda: Path("experiments"))
    logs_dir: Path = field(default_factory=lambda: Path("logs"))
    
    def __post_init__(self):
        """Post-initialization setup"""
        # Ensure paths are Path objects
        self.data_dir = Path(self.data_dir)
        self.models_dir = Path(self.models_dir)
        self.experiments_dir = Path(self.experiments_dir)
        self.logs_dir = Path(self.logs_dir)
        
        # Create directories if they don't exist
        for path in [self.models_dir, self.experiments_dir, self.logs_dir]:
            path.mkdir(parents=True, exist_ok=True)
        
        # Set random seeds for reproducibility
        import tensorflow as tf
        import numpy as np
        import random
        
        tf.random.set_seed(self.random_seed)
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        
        # Configure mixed precision if enabled
        if self.training.use_mixed_precision and tf.config.list_physical_devices('GPU'):
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        import dataclasses
        return dataclasses.asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create config from dictionary"""
        return cls(**config_dict)
    
    def save(self, path: str):
        """Save configuration to YAML file"""
        import yaml
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    @classmethod
    def load(cls, path: str) -> 'Config':
        """Load configuration from YAML file"""
        import yaml
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)