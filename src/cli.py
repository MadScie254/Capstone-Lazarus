"""
Command Line Interface for CAPSTONE-LAZARUS ML Framework
========================================================
"""

import click
import logging
from pathlib import Path
from typing import Optional
import yaml

from src.config import Config
from src.training.trainer import Trainer
from src.data.etl import DataPipeline
from src.models.factory import ModelFactory
# Optional MLflow logger import with graceful fallback
try:
    from src.experiments.mlflow_logger import MLflowLogger  # type: ignore
except Exception:
    MLflowLogger = None  # type: ignore

logger = logging.getLogger(__name__)

@click.group()
@click.option('--config', '-c', type=click.Path(exists=True), help='Config file path')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx, config: Optional[str], verbose: bool):
    """CAPSTONE-LAZARUS ML Framework CLI"""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    if config:
        with open(config, 'r') as f:
            config_dict = yaml.safe_load(f)
        ctx.obj = Config(**config_dict)
    else:
        ctx.obj = Config()

@cli.command()
@click.option('--task', type=click.Choice(['classification', 'regression']), default='classification')
@click.option('--data-path', type=click.Path(exists=True), required=True)
@click.option('--model-name', default='efficient_net')
@click.option('--epochs', default=10, type=int)
@click.option('--batch-size', default=32, type=int)
@click.pass_context
def train(ctx, task: str, data_path: str, model_name: str, epochs: int, batch_size: int):
    """Train a model"""
    config = ctx.obj
    config.task = task
    config.training.epochs = epochs
    config.training.batch_size = batch_size
    
    logger.info(f"Starting training with config: {config}")
    
    # Initialize components
    data_pipeline = DataPipeline(config)
    model_factory = ModelFactory(config)
    trainer = Trainer(config)
    mlflow_logger = MLflowLogger(config) if (MLflowLogger and getattr(config.mlflow, 'enabled', False)) else None
    
    # Load and preprocess data
    train_ds, val_ds, test_ds = data_pipeline.prepare_datasets(data_path)
    
    # Create model
    # Create model using configured input shape to avoid dataset TypeSpec issues
    input_shape = tuple(config.model.input_shape)
    model = model_factory.create_model(model_name, input_shape=input_shape)
    
    # Train model
    history = trainer.train(model, train_ds, val_ds, callbacks=None, mlflow_logger=mlflow_logger)
    
    # Evaluate
    test_metrics = trainer.evaluate(model, test_ds)
    logger.info(f"Test metrics: {test_metrics}")

@cli.command()
@click.option('--model-path', type=click.Path(exists=True), required=True)
@click.option('--data-path', type=click.Path(exists=True), required=True)
@click.option('--output-path', type=click.Path(), required=True)
@click.pass_context
def predict(ctx, model_path: str, data_path: str, output_path: str):
    """Make predictions with a trained model"""
    from src.inference.predictor import Predictor
    from PIL import Image
    import numpy as np
    from pathlib import Path
    
    config = ctx.obj
    predictor = Predictor(config)
    
    # Load model
    if not predictor.load_model(model_path):
        logger.error("Failed to load model for prediction")
        return
    
    data_dir = Path(data_path)
    if not data_dir.exists():
        logger.error(f"Data path not found: {data_dir}")
        return
    
    # Collect image files
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = [p for p in data_dir.rglob("*") if p.suffix.lower() in exts]
    logger.info(f"Found {len(image_files)} images to predict")
    
    from src.inference.predictor import PredictionResult
    results: list[PredictionResult] = []
    for img_path in image_files:
        try:
            img = Image.open(img_path).convert("RGB")
            res = predictor.predict_single(np.array(img))
            # Attach source path into metadata before saving
            res.metadata = {**res.metadata, "source_path": str(img_path)}
            results.append(res)
        except Exception as e:
            logger.error(f"Failed to predict {img_path}: {e}")
    
    # Save predictions
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    predictor.save_predictions(results, str(out_path))
    logger.info(f"Predictions saved to {out_path}")

@cli.command()
@click.option('--port', default=8501, type=int)
@click.option('--host', default='localhost')
def streamlit(port: int, host: str):
    """Launch Streamlit app"""
    import subprocess
    import sys
    
    cmd = [
        sys.executable, '-m', 'streamlit', 'run',
        'app/streamlit_app/main.py',
        '--server.port', str(port),
        '--server.address', host
    ]
    
    logger.info(f"Launching Streamlit app on {host}:{port}")
    subprocess.run(cmd)

@cli.command()
@click.option('--study-name', default='default_study')
@click.option('--n-trials', default=100, type=int)
@click.option('--data-path', type=click.Path(exists=True), required=True)
@click.pass_context
def tune(ctx, study_name: str, n_trials: int, data_path: str):
    """Run hyperparameter tuning"""
    from src.training.tuning import HyperparameterTuner
    
    config = ctx.obj
    tuner = HyperparameterTuner(config)
    best_params = tuner.optimize(study_name, n_trials, data_path)
    logger.info(f"Best parameters: {best_params}")

if __name__ == '__main__':
    cli()