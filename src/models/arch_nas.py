"""
Neural Architecture Search (NAS) for CAPSTONE-LAZARUS
====================================================
"""

import tensorflow as tf
from tensorflow import keras
import optuna
import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Optional
import json
from pathlib import Path
from datetime import datetime

from src.config import Config
from src.models.factory import ModelFactory

logger = logging.getLogger(__name__)

class NASSearchSpace:
    """Defines search spaces for Neural Architecture Search"""
    
    @staticmethod
    def efficient_net_space(trial: optuna.Trial) -> Dict[str, Any]:
        """EfficientNet search space"""
        return {
            "architecture": "efficient_net",
            "variant": trial.suggest_categorical("variant", ["B0", "B1", "B2", "B3"]),
            "pretrained": trial.suggest_categorical("pretrained", [True, False]),
            "dropout_rate": trial.suggest_float("dropout_rate", 0.1, 0.5),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-6, 1e-3, log=True)
        }
    
    @staticmethod
    def mobilenet_space(trial: optuna.Trial) -> Dict[str, Any]:
        """MobileNet search space"""
        return {
            "architecture": "mobilenet",
            "variant": trial.suggest_categorical("variant", ["V1", "V2", "V3Small", "V3Large"]),
            "pretrained": trial.suggest_categorical("pretrained", [True, False]),
            "dropout_rate": trial.suggest_float("dropout_rate", 0.1, 0.5),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-6, 1e-3, log=True)
        }
    
    @staticmethod
    def custom_cnn_space(trial: optuna.Trial) -> Dict[str, Any]:
        """Custom CNN search space"""
        num_blocks = trial.suggest_int("num_blocks", 2, 5)
        filters = []
        for i in range(num_blocks):
            filters.append(trial.suggest_categorical(f"filters_block_{i}", [32, 64, 128, 256, 512]))
        
        return {
            "architecture": "custom_cnn",
            "num_blocks": num_blocks,
            "filters": filters,
            "dropout_rate": trial.suggest_float("dropout_rate", 0.2, 0.6),
            "batch_norm": trial.suggest_categorical("batch_norm", [True, False]),
            "activation": trial.suggest_categorical("activation", ["relu", "swish", "gelu"])
        }
    
    @staticmethod
    def vit_space(trial: optuna.Trial) -> Dict[str, Any]:
        """Vision Transformer search space"""
        return {
            "architecture": "vit",
            "patch_size": trial.suggest_categorical("patch_size", [8, 16, 32]),
            "num_heads": trial.suggest_categorical("num_heads", [4, 8, 12, 16]),
            "num_layers": trial.suggest_int("num_layers", 4, 12),
            "hidden_dim": trial.suggest_categorical("hidden_dim", [256, 512, 768, 1024]),
            "dropout_rate": trial.suggest_float("dropout_rate", 0.1, 0.3)
        }
    
    @staticmethod
    def mlp_space(trial: optuna.Trial) -> Dict[str, Any]:
        """MLP search space"""
        num_layers = trial.suggest_int("num_layers", 2, 6)
        hidden_layers = []
        for i in range(num_layers):
            units = trial.suggest_categorical(f"units_layer_{i}", [64, 128, 256, 512, 1024])
            hidden_layers.append(units)
        
        return {
            "architecture": "mlp",
            "hidden_layers": hidden_layers,
            "dropout_rate": trial.suggest_float("dropout_rate", 0.2, 0.6),
            "batch_norm": trial.suggest_categorical("batch_norm", [True, False])
        }

class NeuralArchitectureSearch:
    """Neural Architecture Search implementation using Optuna"""
    
    def __init__(self, config: Config):
        self.config = config
        self.nas_config = config.nas
        self.model_factory = ModelFactory(config)
        self.study_path = config.experiments_dir / "nas_studies"
        self.study_path.mkdir(parents=True, exist_ok=True)
    
    def search(self, 
               train_dataset: tf.data.Dataset,
               val_dataset: tf.data.Dataset,
               search_space: str = "efficient_net",
               study_name: Optional[str] = None,
               n_trials: Optional[int] = None) -> Dict[str, Any]:
        """
        Perform Neural Architecture Search
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            search_space: Search space name
            study_name: Name of the study
            n_trials: Number of trials to run
            
        Returns:
            Best architecture and results
        """
        if not self.nas_config.enabled:
            logger.warning("NAS is disabled in configuration")
            return {}
        
        if study_name is None:
            study_name = f"nas_{search_space}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if n_trials is None:
            n_trials = self.nas_config.num_trials
        
        logger.info(f"Starting NAS with {n_trials} trials for {search_space} search space")
        
        # Create Optuna study
        study = optuna.create_study(
            direction="maximize",
            study_name=study_name,
            storage=f"sqlite:///{self.study_path / f'{study_name}.db'}"
        )
        
        # Define objective function
        def objective(trial):
            return self._objective_function(
                trial, 
                train_dataset, 
                val_dataset, 
                search_space
            )
        
        # Optimize
        study.optimize(objective, n_trials=n_trials)
        
        # Get best results
        best_params = study.best_params
        best_value = study.best_value
        
        # Save results
        results = {
            "study_name": study_name,
            "search_space": search_space,
            "best_params": best_params,
            "best_value": best_value,
            "n_trials": len(study.trials),
            "timestamp": datetime.now().isoformat()
        }
        
        self._save_nas_results(results)
        
        logger.info(f"NAS completed. Best validation accuracy: {best_value:.4f}")
        logger.info(f"Best parameters: {best_params}")
        
        return results
    
    def _objective_function(self, 
                           trial: optuna.Trial,
                           train_dataset: tf.data.Dataset,
                           val_dataset: tf.data.Dataset,
                           search_space: str) -> float:
        """Objective function for optimization"""
        
        try:
            # Sample architecture parameters
            if search_space == "efficient_net":
                arch_params = NASSearchSpace.efficient_net_space(trial)
            elif search_space == "mobilenet":
                arch_params = NASSearchSpace.mobilenet_space(trial)
            elif search_space == "custom_cnn":
                arch_params = NASSearchSpace.custom_cnn_space(trial)
            elif search_space == "vit":
                arch_params = NASSearchSpace.vit_space(trial)
            elif search_space == "mlp":
                arch_params = NASSearchSpace.mlp_space(trial)
            else:
                raise ValueError(f"Unknown search space: {search_space}")
            
            # Sample training hyperparameters
            learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
            batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
            
            # Update config temporarily
            original_config = self.config
            temp_config = self.config
            temp_config.training.learning_rate = learning_rate
            temp_config.data.batch_size = batch_size
            
            # Update model config with architecture parameters
            for key, value in arch_params.items():
                if hasattr(temp_config.model, key):
                    setattr(temp_config.model, key, value)
            
            # Get input shape from dataset
            sample_batch = next(iter(train_dataset))
            input_shape = sample_batch[0].shape[1:]
            
            # Create model
            model = self.model_factory.create_model(
                arch_params["architecture"],
                input_shape,
                **{k: v for k, v in arch_params.items() if k != "architecture"}
            )
            
            # Train model for limited epochs
            max_epochs = min(self.nas_config.max_epochs, 20)  # Limit epochs for NAS
            
            # Early stopping for efficiency
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=3,
                restore_best_weights=True
            )
            
            # Train model
            history = model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=max_epochs,
                callbacks=[early_stopping],
                verbose=0
            )
            
            # Get best validation accuracy
            val_accuracy = max(history.history['val_accuracy'])
            
            # Optionally apply pruning for unsuccessful trials
            if self.nas_config.pruning_enabled:
                trial.report(val_accuracy, max_epochs)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            # Clean up
            del model
            tf.keras.backend.clear_session()
            
            return val_accuracy
            
        except optuna.TrialPruned:
            raise
        except Exception as e:
            logger.warning(f"Trial failed: {e}")
            return 0.0
    
    def _save_nas_results(self, results: Dict[str, Any]):
        """Save NAS results to file"""
        results_file = self.study_path / f"{results['study_name']}_results.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"NAS results saved to {results_file}")
    
    def create_best_model(self, 
                         nas_results: Dict[str, Any],
                         input_shape: Tuple[int, ...]) -> keras.Model:
        """Create the best model from NAS results"""
        
        best_params = nas_results["best_params"]
        architecture = best_params.get("architecture", "efficient_net")
        
        # Update config with best parameters
        for key, value in best_params.items():
            if hasattr(self.config.model, key):
                setattr(self.config.model, key, value)
            elif hasattr(self.config.training, key):
                setattr(self.config.training, key, value)
        
        # Create model
        model = self.model_factory.create_model(
            architecture,
            input_shape,
            **{k: v for k, v in best_params.items() if k not in ["architecture", "learning_rate", "batch_size"]}
        )
        
        logger.info(f"Created best model from NAS: {architecture}")
        return model
    
    def load_nas_results(self, study_name: str) -> Optional[Dict[str, Any]]:
        """Load NAS results from file"""
        results_file = self.study_path / f"{study_name}_results.json"
        
        if not results_file.exists():
            logger.warning(f"NAS results file not found: {results_file}")
            return None
        
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            logger.info(f"Loaded NAS results from {results_file}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to load NAS results: {e}")
            return None
    
    def list_studies(self) -> List[str]:
        """List available NAS studies"""
        study_files = list(self.study_path.glob("*_results.json"))
        study_names = [f.stem.replace("_results", "") for f in study_files]
        return sorted(study_names)
    
    def get_study_summary(self, study_name: str) -> Optional[Dict[str, Any]]:
        """Get summary of a NAS study"""
        results = self.load_nas_results(study_name)
        
        if results is None:
            return None
        
        return {
            "study_name": results["study_name"],
            "search_space": results["search_space"],
            "best_value": results["best_value"],
            "n_trials": results["n_trials"],
            "timestamp": results["timestamp"],
            "best_architecture": results["best_params"].get("architecture", "unknown")
        }

class ENAS:
    """Efficient Neural Architecture Search using weight sharing"""
    
    def __init__(self, config: Config):
        self.config = config
        self.nas_config = config.nas
    
    def search(self, 
               train_dataset: tf.data.Dataset,
               val_dataset: tf.data.Dataset,
               **kwargs) -> Dict[str, Any]:
        """
        Perform ENAS (placeholder implementation)
        
        This is a simplified version. Full ENAS implementation would require
        a supernet and controller network.
        """
        logger.info("ENAS search started (simplified implementation)")
        
        # For now, this is a placeholder that delegates to regular NAS
        nas = NeuralArchitectureSearch(self.config)
        return nas.search(train_dataset, val_dataset, **kwargs)

# Utility functions
def run_nas_search(config: Config,
                   train_dataset: tf.data.Dataset,
                   val_dataset: tf.data.Dataset,
                   search_space: str = "efficient_net",
                   n_trials: int = 100) -> Dict[str, Any]:
    """Convenience function to run NAS"""
    nas = NeuralArchitectureSearch(config)
    return nas.search(train_dataset, val_dataset, search_space, n_trials=n_trials)