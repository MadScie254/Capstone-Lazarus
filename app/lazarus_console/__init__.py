"""
Lazarus Console - Single-file immersive Streamlit mission control for plant disease diagnostics.
"""

from __future__ import annotations

import base64
import dataclasses
import io
import json
import math
import os
import sys
import textwrap
import time
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple, cast

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

try:  # pragma: no cover - local utilities
    from app.lazarus_console.utils.model_manager import ModelManager
except Exception:  # pragma: no cover
    ModelManager = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    if torch is not None:
        import torch.nn as nn  # type: ignore
    else:  # pragma: no cover
        nn = None  # type: ignore[assignment]
except ImportError:  # pragma: no cover
    nn = None  # type: ignore[assignment]

try:  # pragma: no cover
    from torchvision import models, transforms  # type: ignore
except ImportError:  # pragma: no cover
    models = None  # type: ignore
    transforms = None  # type: ignore

try:  # pragma: no cover
    import onnxruntime as ort  # type: ignore
except ImportError:  # pragma: no cover
    ort = None  # type: ignore

try:  # pragma: no cover
    from sklearn.metrics import (  # type: ignore
        confusion_matrix,
        f1_score,
        precision_recall_curve,
        recall_score,
    )
except ImportError:  # pragma: no cover
    confusion_matrix = f1_score = precision_recall_curve = recall_score = None  # type: ignore

if TYPE_CHECKING:  # pragma: no cover
    from torch import Tensor
    from torch.nn import Module
    from onnxruntime import InferenceSession  # type: ignore
else:
    Tensor = Any
    Module = Any
    InferenceSession = Any


def get_project_root() -> Path:
    current = Path(__file__).resolve()
    for candidate in current.parents:
        if (candidate / "config.yaml").exists() or (candidate / "requirements.txt").exists():
            return candidate
    return current.parent


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = get_project_root()
DATA_DIR = PROJECT_ROOT / "data"
MODEL_EXPORT_DIR = PROJECT_ROOT / "models" / "exports"
CLASS_NAMES_PATH = PROJECT_ROOT / "models" / "class_names.json"
EXPERIMENTS_INDEX_PATH = PROJECT_ROOT / "experiments.csv"
MODEL_EXPORT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_MANAGER: Optional[ModelManager] = None
MODEL_SPECS: Dict[str, Dict[str, Any]] = {}
if ModelManager is not None:
    try:
        MODEL_MANAGER = ModelManager(PROJECT_ROOT)
        MODEL_SPECS = MODEL_MANAGER.get_console_model_specs()
    except Exception:  # pragma: no cover - registry errors fallback to defaults
        MODEL_SPECS = {}


# --------------------------------------------------------------------------------------------------
# Adaptive Theming
# --------------------------------------------------------------------------------------------------

THEMES: Dict[str, Dict[str, str]] = {
    "dark": {
        "background": "#0a0d16",
        "surface": "#1a1f2e",
        "card": "linear-gradient(145deg, rgba(45,56,99,0.85), rgba(22,26,46,0.92))",
        "text_primary": "#f8faff",
        "text_secondary": "#b8bbc8",
        "accent": "#4ade80",
        "accent_soft": "rgba(74, 222, 128, 0.18)",
        "warning": "#fbbf24",
        "error": "#f87171",
        "success": "#34d399",
        "border": "rgba(255,255,255,0.12)",
        "shadow": "0 25px 50px rgba(0,0,0,0.45)",
        "glow": "0 0 30px rgba(74, 222, 128, 0.3)",
    },
    "light": {
        "background": "#f8fafc",
        "surface": "#ffffff",
        "card": "linear-gradient(145deg, rgba(251,253,255,0.95), rgba(240,248,255,0.95))",
        "text_primary": "#1e293b",
        "text_secondary": "#475569",
        "accent": "#2563eb",
        "accent_soft": "rgba(37,99,235,0.15)",
        "warning": "#d97706",
        "error": "#dc2626",
        "success": "#059669",
        "border": "rgba(30, 41, 59, 0.12)",
        "shadow": "0 20px 40px rgba(15,30,80,0.25)",
        "glow": "0 0 25px rgba(37,99,235,0.2)",
    },
    "neon": {
        "background": "#050508",
        "surface": "#0f0f17",
        "card": "linear-gradient(145deg, rgba(139,69,255,0.15), rgba(59,130,246,0.12))",
        "text_primary": "#e5e7eb",
        "text_secondary": "#9ca3af",
        "accent": "#8b5cf6",
        "accent_soft": "rgba(139,92,246,0.2)",
        "warning": "#f59e0b",
        "error": "#ef4444",
        "success": "#10b981",
        "border": "rgba(139,92,246,0.3)",
        "shadow": "0 25px 50px rgba(139,92,246,0.4)",
        "glow": "0 0 40px rgba(139,92,246,0.6)",
    },
}


def build_model_options() -> Dict[str, Dict[str, Any]]:
    """Build model options with integration to trained models."""
    if models is None:
        return {}

    tv_models = models
    
    # Enhanced model specifications with trained model integration
    enhanced_options = {
        "efficientnet_b0": {
            "label": "🚀 EfficientNet-B0 (Your Trained Model)",
            "torch_builder": lambda: tv_models.efficientnet_b0(weights=tv_models.EfficientNet_B0_Weights.DEFAULT),
            "input_size": 160,  # Match your training size
            "onnx_filename": "efficientnet_b0.onnx",
            "ensemble_default_weight": 1.0,
            "accuracy_range": "70-80%",
            "speed": "Fast",
            "memory": "Low",
            "trained": True,
        },
        "mobilenet_v3_small": {
            "label": "📱 MobileNetV3-Small (Backup)",
            "torch_builder": lambda: tv_models.mobilenet_v3_small(weights=tv_models.MobileNet_V3_Small_Weights.DEFAULT),
            "input_size": 224,
            "onnx_filename": "mobilenet_v3_small.onnx",
            "ensemble_default_weight": 0.8,
            "accuracy_range": "65-75%",
            "speed": "Very Fast",
            "memory": "Very Low",
            "trained": False,
        },
        "resnet18": {
            "label": "🔧 ResNet-18 (Baseline)",
            "torch_builder": lambda: tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT),
            "input_size": 224,
            "onnx_filename": "resnet18.onnx", 
            "ensemble_default_weight": 0.6,
            "accuracy_range": "60-70%",
            "speed": "Medium",
            "memory": "Medium",
            "trained": False,
        },
    }

    # Check for existing trained models and update specifications
    if EXPERIMENTS_INDEX_PATH.exists():
        try:
            import pandas as pd
            df = pd.read_csv(EXPERIMENTS_INDEX_PATH)
            
            for model_key in enhanced_options.keys():
                model_runs = df[
                    (df["model_name"].str.contains(model_key, case=False, na=False)) |
                    (df["backbone"].str.contains(model_key, case=False, na=False))
                ]
                
                if not model_runs.empty:
                    # Get best run metrics
                    best_run = model_runs.sort_values("val_macro_f1", ascending=False).iloc[0]
                    accuracy = float(best_run["val_accuracy"]) * 100
                    f1_score = float(best_run["val_macro_f1"]) * 100
                    
                    enhanced_options[model_key]["trained"] = True
                    enhanced_options[model_key]["actual_accuracy"] = f"{accuracy:.1f}%"
                    enhanced_options[model_key]["f1_score"] = f"{f1_score:.1f}%"
                    enhanced_options[model_key]["label"] = f"🎯 {enhanced_options[model_key]['label'].split('(')[0].strip()} (Trained: {accuracy:.1f}%)"
                    
        except Exception as e:
            print(f"Warning: Could not load experiment results: {e}")
    
    return enhanced_options


MODEL_OPTIONS: Dict[str, Dict[str, Any]] = build_model_options()
DEFAULT_ENSEMBLE_WEIGHTS = {key: cfg.get("ensemble_default_weight", 1.0) for key, cfg in MODEL_OPTIONS.items()}


def _format_timestamp(ts) -> str:
    """Format timestamp for display."""
    if pd.isna(ts):
        return "Unknown"
    if isinstance(ts, str):
        try:
            ts = pd.to_datetime(ts)
        except:
            return ts
    return ts.strftime("%Y-%m-%d %H:%M")


def ensure_session_state() -> None:
    defaults = {
        "theme": "dark",
        "selected_model": "efficientnet_b0",
        "precision_mode": "AMP",
        "enable_ensemble": False,
        "confidence_threshold": 0.55,
        "ensemble_weights": DEFAULT_ENSEMBLE_WEIGHTS.copy(),
        "warmup_tracker": {},
        "inference_history": [],
        "explainability_cache": {},
        "current_section": "Home",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if MODEL_OPTIONS:
        if st.session_state.selected_model not in MODEL_OPTIONS:
            st.session_state.selected_model = next(iter(MODEL_OPTIONS))
        if set(st.session_state.ensemble_weights.keys()) != set(MODEL_OPTIONS.keys()):
            weights = DEFAULT_ENSEMBLE_WEIGHTS.copy()
            st.session_state.ensemble_weights = weights


def inject_theme(theme_key: str) -> None:
    palette = THEMES[theme_key]
    css = f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Keyframe Animations */
        @keyframes fadeInUp {{
            from {{ opacity: 0; transform: translateY(30px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        @keyframes progressAnimation {{
            from {{ stroke-dashoffset: 283; }}
            to {{ stroke-dashoffset: var(--target-offset); }}
        }}
        
        @keyframes barFillAnimation {{
            from {{ width: 0%; }}
            to {{ width: var(--target-width); }}
        }}
        
        @keyframes pulse {{
            0% {{ transform: scale(1); }}
            50% {{ transform: scale(1.05); }}
            100% {{ transform: scale(1); }}
        }}
        
        @keyframes glow {{
            0%, 100% {{ text-shadow: 0 0 10px {palette['accent']}; }}
            50% {{ text-shadow: 0 0 20px {palette['accent']}, 0 0 30px {palette['accent']}; }}
        }}
        
        /* Base Styles */
        body, .stApp {{
            background: {palette['background']} !important;
            color: {palette['text_primary']} !important;
            font-family: 'Inter', sans-serif !important;
        }}
        
        /* Card Styles */
        .lazarus-card {{
            background: {palette['card']};
            border-radius: 20px;
            padding: 2rem;
            border: 1px solid {palette['border']};
            box-shadow: {palette['shadow']};
            transition: all 0.3s ease;
            animation: fadeInUp 0.6s ease-out;
        }}
        
        .animated-card {{
            cursor: pointer;
        }}
        
        .animated-card:hover {{
            transform: translateY(-8px) scale(1.02);
            box-shadow: {palette['glow']};
        }}
        
        .animated-card.pulse {{
            animation: pulse 0.6s ease-in-out;
        }}
        
        /* Model Status Cards */
        .model-status-card {{
            background: {palette['card']};
            border-radius: 16px;
            padding: 1.5rem;
            border: 2px solid {palette['border']};
            margin: 0.5rem 0;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }}
        
        .trained-model {{
            border-color: {palette['success']};
            background: linear-gradient(145deg, {palette['card']}, rgba(52, 211, 153, 0.1));
        }}
        
        .pretrained-model {{
            border-color: {palette['accent_soft']};
        }}
        
        .model-header {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 0.5rem;
        }}
        
        .model-icon {{
            font-size: 1.25rem;
        }}
        
        .model-name {{
            font-weight: 600;
            color: {palette['text_primary']};
        }}
        
        .model-status {{
            font-weight: 500;
            margin: 0.25rem 0;
        }}
        
        .model-accuracy {{
            color: {palette['text_secondary']};
            font-size: 0.9rem;
        }}
        
        .model-badge {{
            position: absolute;
            top: 0.5rem;
            right: 0.5rem;
            background: {palette['accent']};
            color: white;
            padding: 0.25rem 0.5rem;
            border-radius: 8px;
            font-size: 0.7rem;
            font-weight: 600;
        }}
        
        /* Progress Rings */
        .progress-ring-container {{
            position: relative;
            display: inline-block;
            margin: 1rem;
        }}
        
        .progress-ring {{
            transform: rotate(-90deg);
        }}
        
        .progress-ring-bg {{
            fill: transparent;
            stroke: {palette['border']};
            stroke-width: 8;
        }}
        
        .progress-ring-fill {{
            fill: transparent;
            stroke-width: 8;
            stroke-linecap: round;
            transition: stroke-dashoffset 0.5s ease;
        }}
        
        .progress-ring-text {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            text-align: center;
        }}
        
        .progress-percentage {{
            font-size: 1.5rem;
            font-weight: 700;
            color: {palette['accent']};
        }}
        
        .progress-label {{
            font-size: 0.8rem;
            color: {palette['text_secondary']};
            margin-top: 0.25rem;
        }}
        
        /* Confidence Bars */
        .confidence-bars-container {{
            margin: 1rem 0;
        }}
        
        .confidence-bar-item {{
            display: flex;
            align-items: center;
            gap: 1rem;
            margin: 0.75rem 0;
        }}
        
        .confidence-label {{
            min-width: 120px;
            font-weight: 500;
            color: {palette['text_primary']};
        }}
        
        .confidence-bar-bg {{
            flex: 1;
            height: 8px;
            background: {palette['border']};
            border-radius: 4px;
            overflow: hidden;
        }}
        
        .confidence-bar-fill {{
            height: 100%;
            border-radius: 4px;
            transition: width 0.5s ease;
        }}
        
        .confidence-value {{
            min-width: 60px;
            text-align: right;
            font-weight: 600;
            color: {palette['accent']};
        }}
        
        /* Metric Styles */
        .metric-header {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 1rem;
        }}
        
        .metric-icon {{
            font-size: 1.25rem;
        }}
        
        .metric-value {{
            font-size: 2.8rem;
            font-weight: 700;
            color: {palette['accent']};
            margin: 0.5rem 0;
        }}
        
        .glow-text {{
            animation: glow 3s ease-in-out infinite;
        }}
        
        .metric-label {{
            color: {palette['text_secondary']};
            font-size: 1rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-weight: 500;
        }}
        
        .metric-delta {{
            color: {palette['success']};
            font-size: 0.9rem;
            font-weight: 500;
            margin-top: 0.5rem;
        }}
        
        /* Streamlit Component Styling */
        .stSelectbox > div > div {{
            background: {palette['surface']} !important;
            border: 1px solid {palette['border']} !important;
            border-radius: 12px !important;
        }}
        
        .stButton > button {{
            background: {palette['accent']} !important;
            color: white !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 0.75rem 2rem !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
        }}
        
        .stButton > button:hover {{
            background: {palette['accent']} !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 10px 20px rgba(0,0,0,0.2) !important;
        }}
        
        /* File Uploader */
        .uploadedFile {{
            background: {palette['surface']} !important;
            border: 2px dashed {palette['accent_soft']} !important;
            border-radius: 16px !important;
            padding: 2rem !important;
            text-align: center !important;
        }}
        
        /* Sidebar */
        .css-1d391kg {{
            background: {palette['surface']} !important;
        }}
        
        /* Success/Warning Messages */
        .stSuccess {{
            background: linear-gradient(90deg, {palette['success']}, {palette['success']}aa) !important;
            border-radius: 12px !important;
        }}
        
        .stWarning {{
            background: linear-gradient(90deg, {palette['warning']}, {palette['warning']}aa) !important;
            border-radius: 12px !important;
        }}
        
        .stError {{
            background: linear-gradient(90deg, {palette['error']}, {palette['error']}aa) !important;
            border-radius: 12px !important;
            border-left: 4px solid {palette['accent']};
        }}
        
        .threshold-highlight {{
            background: {palette['accent_soft']};
            border-radius: 12px;
            padding: 0.75rem 1rem;
        }}
        
        .section-divider {{
            height: 1px;
            background: {palette['border']};
            margin: 2.5rem 0 1.5rem 0;
        }}
        
        .skeleton {{
            background: linear-gradient(90deg, rgba(255,255,255,0) 0%, rgba(255,255,255,0.25) 50%, rgba(255,255,255,0) 100%);
            background-size: 200% 100%;
            animation: shimmer 1.4s infinite;
        }}
        @keyframes shimmer {{
            0% {{ background-position: -200% 0; }}
            100% {{ background-position: 200% 0; }}
        }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# --------------------------------------------------------------------------------------------------
# Model Loading Utilities
# --------------------------------------------------------------------------------------------------


def get_device() -> Any:
    if torch is None:
        raise RuntimeError("PyTorch is required for inference but is not installed.")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_best_checkpoint_for_model(model_key: str) -> Optional[Path]:
    """Find the best checkpoint for a model from experiments.csv."""
    if not EXPERIMENTS_INDEX_PATH.exists():
        return None
    
    try:
        import pandas as pd
        df = pd.read_csv(EXPERIMENTS_INDEX_PATH)
        
        # Filter for this model (match by name or backbone)
        model_runs = df[
            (df["model_name"].str.contains(model_key, case=False, na=False)) |
            (df["backbone"].str.contains(model_key, case=False, na=False))
        ]
        
        if model_runs.empty:
            return None
        
        # Sort by F1 score, get best
        model_runs = model_runs.sort_values("val_macro_f1", ascending=False)
        best_run = model_runs.iloc[0]
        
        checkpoint_path_str = best_run.get("best_checkpoint_path")
        if pd.isna(checkpoint_path_str) or not checkpoint_path_str:
            return None
        
        checkpoint_path = PROJECT_ROOT / checkpoint_path_str
        if checkpoint_path.exists():
            return checkpoint_path
        
        return None
    except Exception as e:
        print(f"Error loading checkpoint for {model_key}: {e}")
        return None


@st.cache_resource(show_spinner=False)
def load_torch_model(model_key: str, use_trained: bool = True) -> Optional[Module]:
    """Load PyTorch model, preferring trained checkpoints from experiments.csv."""
    if torch is None or models is None:
        st.toast("PyTorch not available - install torch and torchvision.", icon="⚠️")
        return None
    
    config = MODEL_OPTIONS[model_key]
    num_classes = len(load_class_names())
    
    # Try to load trained checkpoint first
    checkpoint_path = None
    if use_trained:
        checkpoint_path = get_best_checkpoint_for_model(model_key)
    
    if checkpoint_path and checkpoint_path.exists():
        # Load trained model using src infrastructure
        try:
            from src.model_factory_torch import get_model
            
            # Infer backbone name from model_key
            backbone = model_key.replace("_", "")  # efficientnet_b0 -> efficientnetb0
            
            model = get_model(
                backbone=model_key,
                num_classes=num_classes,
                pretrained=False,
                dropout_rate=0.3
            )
            
            # Load trained weights
            state_dict = torch.load(checkpoint_path, map_location=get_device())
            model.load_state_dict(state_dict)
            model.eval()
            model.to(get_device())
            
            st.toast(f"✓ Loaded trained {MODEL_OPTIONS[model_key]['label']} from checkpoint", icon="🎯")
            return model
            
        except Exception as e:
            st.warning(f"Failed to load trained checkpoint: {e}. Falling back to pretrained ImageNet weights.")
    
    # Fallback: load pretrained ImageNet model
    model = config["torch_builder"]()
    model = adapt_model_head(model, model_key)
    model.eval()
    model.to(get_device())
    
    if not checkpoint_path:
        st.info(f"No trained checkpoint found for {model_key}. Using ImageNet pretrained weights. Train a model first!")
    
    return model


def adapt_model_head(model: Module, model_key: str) -> Module:
    if torch is None or nn is None:
        return model

    try:
        num_classes = len(load_class_names())
    except Exception:
        return model

    if num_classes <= 0:
        return model

    def _replace_linear(module: Any, in_features: Optional[int]) -> Optional[nn.Linear]:
        if in_features is None:
            return None
        linear = nn.Linear(in_features, num_classes)
        nn.init.xavier_uniform_(linear.weight)
        if linear.bias is not None:
            nn.init.zeros_(linear.bias)
        return linear

    replaced = False

    classifier = getattr(model, "classifier", None)
    if isinstance(classifier, nn.Sequential) and classifier:
        layers = list(classifier.children())
        last_layer = layers[-1]
        in_features = getattr(last_layer, "in_features", None)
        out_features = getattr(last_layer, "out_features", None)
        if out_features != num_classes:
            new_layer = _replace_linear(last_layer, in_features)
            if new_layer is not None:
                layers[-1] = new_layer
                model.classifier = nn.Sequential(*layers)
                replaced = True
    elif hasattr(classifier, "out_features") and hasattr(classifier, "in_features"):
        out_features = getattr(classifier, "out_features")
        in_features = getattr(classifier, "in_features")
        if out_features != num_classes:
            new_layer = _replace_linear(classifier, in_features)
            if new_layer is not None:
                model.classifier = new_layer
                replaced = True

    fc = getattr(model, "fc", None)
    if hasattr(fc, "out_features") and hasattr(fc, "in_features"):
        if getattr(fc, "out_features") != num_classes:
            new_fc = _replace_linear(fc, getattr(fc, "in_features", None))
            if new_fc is not None:
                model.fc = new_fc
                replaced = True

    if replaced:
        st.toast(f"Retuned final layer for {MODEL_OPTIONS[model_key]['label']} to {num_classes} classes", icon="🧬")

    return model


def ensure_onnx_model(model_key: str) -> Optional[Path]:
    if torch is None:
        return None
    assert torch is not None
    config = MODEL_OPTIONS[model_key]
    onnx_path = MODEL_EXPORT_DIR / config["onnx_filename"]
    if onnx_path.exists():
        return onnx_path

    model = load_torch_model(model_key)
    if model is None:
        return None

    dummy_input = (torch.randn(1, 3, config["input_size"], config["input_size"], device=get_device()),)
    assert torch is not None
    assert isinstance(dummy_input[0], torch.Tensor)
    try:
        torch.onnx.export(  # type: ignore[call-overload]
            model,
            dummy_input,
            onnx_path,
            input_names=["input"],
            output_names=["logits"],
            opset_version=13,
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        )
        st.toast(f"Exported {config['label']} to ONNX", icon="🧩")
    except Exception as exc:  # pragma: no cover - export path
        st.toast(f"ONNX export failed for {config['label']}: {exc}", icon="⚠️")
        return None

    return onnx_path


@st.cache_resource(show_spinner=False)
def load_onnx_session(model_key: str) -> Optional[InferenceSession]:
    if ort is None:
        st.toast("ONNX Runtime not available", icon="⚠️")
        return None
    onnx_path = ensure_onnx_model(model_key)
    if onnx_path is None:
        return None
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    try:
        session = ort.InferenceSession(str(onnx_path), providers=providers)  # type: ignore[arg-type]
        return session
    except Exception:
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])  # type: ignore[arg-type]
        return session


def get_preprocess_transform(input_size: int) -> Any:
    if transforms is None:
        raise RuntimeError("TorchVision is required for preprocessing but is not installed.")
    assert transforms is not None
    return transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


@dataclass
class InferenceResult:
    image: Image.Image
    probs: np.ndarray
    logits: np.ndarray
    labels: List[str]
    backend: str
    model_key: str
    latency_ms: float


def chunked(iterable: Iterable[Any], size: int) -> Iterable[List[Any]]:
    chunk: List[Any] = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def warmup_once(model_key: str, backend: str) -> None:
    tracker: Dict[str, Dict[str, bool]] = st.session_state.setdefault("warmup_tracker", {})
    if tracker.get(model_key, {}).get(backend):
        return

    tracker.setdefault(model_key, {})[backend] = True
    config = MODEL_OPTIONS[model_key]
    dummy = Image.new("RGB", (config["input_size"], config["input_size"]))
    _ = run_single_backend_inference([dummy], model_key, backend, use_amp=True)


def run_single_backend_inference(
    images: List[Image.Image],
    model_key: str,
    backend: str,
    use_amp: bool,
    ensemble_weights: Optional[Dict[str, float]] = None,
) -> InferenceResult:
    if torch is None or transforms is None:
        raise RuntimeError("PyTorch and TorchVision are required for inference but are not installed.")
    assert torch is not None and transforms is not None
    labels = load_class_names()
    config = MODEL_OPTIONS[model_key]
    input_size = config["input_size"]
    transform = get_preprocess_transform(input_size)
    device = get_device()

    tensors = [transform(img).unsqueeze(0) for img in images]
    logits_list: List[np.ndarray] = []
    start = time.perf_counter()

    if backend == "PyTorch":
        model = load_torch_model(model_key)
        if model is None:
            raise RuntimeError("Model not available")
        model.to(device)
        model.eval()
        with torch.no_grad():
            for sub_batch in chunked(tensors, 8):
                sub_tensor = torch.cat(sub_batch, dim=0).to(device)
                with torch.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
                    outputs = model(sub_tensor)
                logits_list.append(outputs.detach().cpu().numpy())
    else:
        session = load_onnx_session(model_key)
        if session is None:
            raise RuntimeError("ONNX session not available")
        for sub_batch in chunked(tensors, 8):
            sub_tensor = torch.cat(sub_batch, dim=0).numpy()
            ort_inputs = {session.get_inputs()[0].name: sub_tensor}
            logits = session.run(None, ort_inputs)[0]
            logits_list.append(np.asarray(logits))

    logits = np.concatenate(logits_list, axis=0)
    latency_ms = (time.perf_counter() - start) * 1000
    probs = softmax(logits)
    
    # Log inference telemetry
    try:
        from src.telemetry import log_inference
        
        for idx, (prob_vec, img) in enumerate(zip(probs, images)):
            top1_idx = int(np.argmax(prob_vec))
            top1_label = labels[top1_idx]
            top1_conf = float(prob_vec[top1_idx])
            
            # Try to get run_id from checkpoint
            run_id = None
            checkpoint_path = get_best_checkpoint_for_model(model_key)
            if checkpoint_path:
                run_id = checkpoint_path.parent.name
            
            log_inference(
                run_id=run_id,
                model_path=f"console/{model_key}/{backend}",
                image_name=f"upload_{idx}",
                top1_label=top1_label,
                top1_confidence=top1_conf,
                latency_ms=latency_ms / len(images),  # Per-image latency
            )
    except Exception as e:
        # Telemetry is non-critical, don't break inference
        print(f"Telemetry logging failed: {e}")

    return InferenceResult(
        image=images[0] if len(images) == 1 else images[0],
        probs=probs,
        logits=logits,
        labels=labels,
        backend=backend,
        model_key=model_key,
        latency_ms=latency_ms,
    )


def blend_logits(results: List[InferenceResult], weights: Dict[str, float]) -> InferenceResult:
    logits_stack = np.stack([res.logits * weights[res.model_key] for res in results], axis=0)
    blended_logits = np.sum(logits_stack, axis=0) / sum(weights.values())
    template = results[0]
    return InferenceResult(
        image=template.image,
        probs=softmax(blended_logits),
        logits=blended_logits,
        labels=template.labels,
        backend="Ensemble",
        model_key="ensemble",
        latency_ms=max(res.latency_ms for res in results),
    )


# --------------------------------------------------------------------------------------------------
# Dataset Utilities for Metrics & Compare View
# --------------------------------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def load_class_names() -> List[str]:
    if CLASS_NAMES_PATH.exists():
        with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as fp:
            return json.load(fp)
    return sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])


@dataclass
class DatasetManifest:
    frame: pd.DataFrame
    sample_paths: Dict[str, List[Path]]


@st.cache_data(show_spinner=True)
def build_dataset_manifest(sample_per_class: int = 12) -> DatasetManifest:
    class_names = load_class_names()
    records: List[Dict[str, Any]] = []
    sample_paths: Dict[str, List[Path]] = {}
    for cls in class_names:
        cls_dir = DATA_DIR / cls
        if not cls_dir.exists():
            continue
        images = sorted([p for p in cls_dir.glob("**/*.jpg")]) + sorted(
            [p for p in cls_dir.glob("**/*.png")]
        )
        if not images:
            continue
        sample_paths[cls] = images[: sample_per_class]
        for path in sample_paths[cls]:
            records.append({"path": str(path), "label": cls})
    frame = pd.DataFrame(records)
    return DatasetManifest(frame=frame, sample_paths=sample_paths)


@st.cache_data(show_spinner=False)
def load_experiments_index(limit: int = 6) -> pd.DataFrame:
    if not EXPERIMENTS_INDEX_PATH.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(EXPERIMENTS_INDEX_PATH)
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return df

    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce")
    numeric_cols = [
        "val_macro_f1",
        "val_macro_recall",
        "val_accuracy",
        "params_count",
        "batch_size",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


    df = df.sort_values("timestamp_utc", ascending=False)
    if limit:
        df = df.head(limit)
    return df.reset_index(drop=True)


def _format_timestamp(value: Any) -> str:
    if isinstance(value, pd.Timestamp):
        ts = value.tz_localize(None) if value.tzinfo else value
        return ts.strftime("%Y-%m-%d %H:%M")
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed.strftime("%Y-%m-%d %H:%M")
        except ValueError:
            return value
    return "—"


def build_home_metrics() -> Dict[str, Any]:
    class_names = load_class_names()
    fallback = {
        "macro_f1": float("nan"),
        "per_class_recall": np.array([]),
        "latency": float("nan"),
        "model_size": "—",
        "checkpoints": [],
        "latest_run": None,
    }

    def _safe_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            try:
                if math.isnan(value):
                    return None
            except TypeError:
                return None
            return float(value)
        if isinstance(value, str):
            trimmed = value.strip()
            if not trimmed:
                return None
            try:
                cast_value = float(trimmed)
            except ValueError:
                return None
            return cast_value if not math.isnan(cast_value) else None
        if pd.isna(value):  # type: ignore[arg-type]
            return None
        try:
            cast_value = float(value)
            return cast_value if not math.isnan(cast_value) else None
        except (TypeError, ValueError):
            return None

    df = load_experiments_index(limit=6)
    if df.empty:
        return fallback

    latest = df.iloc[0]
    macro_f1 = _safe_float(latest.get("val_macro_f1"))
    macro_recall = _safe_float(latest.get("val_macro_recall"))
    if macro_recall is not None and class_names:
        per_class_recall = np.full(len(class_names), macro_recall)
    elif macro_recall is not None:
        per_class_recall = np.array([macro_recall])
    else:
        per_class_recall = np.array([])

    params_count = _safe_float(latest.get("params_count"))
    if params_count is not None:
        approx_mb = params_count * 4 / (1024 ** 2)
        model_size = f"~{approx_mb:.1f} MB"
    else:
        model_size = "—"

    latency_value = _safe_float(latest.get("latency_ms")) or float("nan")

    checkpoints: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        f1_value = _safe_float(row.get("val_macro_f1"))
        f1_metric = f"F1 {f1_value:.3f}" if f1_value is not None else "F1 —"
        acc_value = _safe_float(row.get("val_accuracy"))
        acc_metric = f"Acc {acc_value:.3f}" if acc_value is not None else None
        notes = (row.get("notes") or "").strip()
        checkpoints.append(
            {
                "timestamp": _format_timestamp(row.get("timestamp_utc")),
                "model": row.get("model_name", "—"),
                "metric": f1_metric,
                "secondary": acc_metric,
                "notes": notes,
                "run_id": row.get("run_id"),
            }
        )

    latest_run = {
        "run_id": latest.get("run_id"),
        "timestamp": _format_timestamp(latest.get("timestamp_utc")),
        "model_name": latest.get("model_name"),
        "backbone": latest.get("backbone"),
        "val_accuracy": _safe_float(latest.get("val_accuracy")),
        "val_macro_f1": macro_f1,
        "notes": (latest.get("notes") or "").strip(),
    }

    return {
        "macro_f1": macro_f1 if macro_f1 is not None else float("nan"),
        "per_class_recall": per_class_recall,
        "latency": latency_value,
        "model_size": model_size,
        "checkpoints": checkpoints,
        "latest_run": latest_run,
    }


@st.cache_data(show_spinner=True)
def cached_model_metrics(model_key: str, backend: str, threshold: float) -> Dict[str, Any]:
    if f1_score is None or recall_score is None or confusion_matrix is None or precision_recall_curve is None:
        raise RuntimeError("scikit-learn is required for metric calculations but is not installed.")
    manifest = build_dataset_manifest()
    if manifest.frame.empty:
        raise RuntimeError("Dataset manifest empty - ensure data directory is populated.")

    images = [Image.open(path).convert("RGB") for path in manifest.frame["path"]]
    result = run_single_backend_inference(images, model_key, backend, use_amp=True)
    probs = result.probs
    labels = manifest.frame["label"].tolist()
    class_names = load_class_names()
    label_to_idx = {name: idx for idx, name in enumerate(class_names)}
    true_indices = np.array([label_to_idx[label] for label in labels])
    preds_indices = np.argmax(probs, axis=1)

    macro_f1 = f1_score(true_indices, preds_indices, average="macro")
    per_class_recall = recall_score(true_indices, preds_indices, average=None, zero_division=0)

    conf_matrix = confusion_matrix(true_indices, preds_indices)

    confidences = np.max(probs, axis=1)
    threshold_mask = confidences >= threshold
    tp = np.sum((preds_indices == true_indices) & threshold_mask)
    fp = np.sum((preds_indices != true_indices) & threshold_mask)
    fn = np.sum((preds_indices != true_indices) & ~threshold_mask)
    tn = np.sum((preds_indices == true_indices) & ~threshold_mask)

    precision, recall, _ = precision_recall_curve(true_indices == preds_indices, confidences)

    # Calibration: reliability diagram + Expected Calibration Error (ECE)
    bins = np.linspace(0.0, 1.0, 11)
    binids = np.digitize(confidences, bins) - 1
    bin_sums = np.bincount(binids, weights=confidences, minlength=len(bins))
    bin_true = np.bincount(binids, weights=(preds_indices == true_indices), minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))
    nonzero = bin_total > 0
    avg_confidence = np.where(nonzero, bin_sums / bin_total, 0)
    avg_accuracy = np.where(nonzero, bin_true / bin_total, 0)
    ece = np.sum(bin_total[nonzero] / np.sum(bin_total) * np.abs(avg_confidence[nonzero] - avg_accuracy[nonzero]))

    return {
        "macro_f1": macro_f1,
        "per_class_recall": per_class_recall,
        "confusion_matrix": conf_matrix,
        "precision_curve": precision,
        "recall_curve": recall,
        "ece": ece,
        "avg_confidence": avg_confidence,
        "avg_accuracy": avg_accuracy,
        "threshold_counts": {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)},
    }


# --------------------------------------------------------------------------------------------------
# Grad-CAM Implementation
# --------------------------------------------------------------------------------------------------


def generate_gradcam(
    model: Module,
    image_tensor: Any,
    target_layer_name: str,
    target_class: int,
) -> np.ndarray:
    if torch is None:
        raise RuntimeError("PyTorch is required for Grad-CAM but is not installed.")
    torch_module = torch
    assert torch_module is not None
    gradients: List[Any] = []
    activations: List[Any] = []

    def forward_hook(_module, _input, output):
        activations.append(output.detach())

    def backward_hook(_module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    target_layer = dict(model.named_modules())[target_layer_name]
    handle_forward = target_layer.register_forward_hook(forward_hook)
    handle_backward = target_layer.register_full_backward_hook(backward_hook)

    model.zero_grad(set_to_none=True)
    output = model(image_tensor)
    loss = output[0, target_class]
    loss.backward()

    grad = gradients[-1]
    activation = activations[-1]
    weights = torch_module.mean(grad, dim=(2, 3), keepdim=True)
    cam = torch_module.relu(torch_module.sum(weights * activation, dim=1)).squeeze().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    handle_forward.remove()
    handle_backward.remove()
    return cam


# --------------------------------------------------------------------------------------------------
# UI Helpers
# --------------------------------------------------------------------------------------------------


def toast_success(message: str) -> None:
    st.toast(message, icon="✅")


def toast_warning(message: str) -> None:
    st.toast(message, icon="⚠️")


def format_latency(latency_ms: float) -> str:
    if latency_ms < 1.0:
        return f"{latency_ms * 1000:.1f} μs"
    return f"{latency_ms:.1f} ms"


def render_metric_card(title: str, value: str, delta: Optional[str] = None, icon: str = "📊") -> None:
    """Render a beautiful animated metric card."""
    st.markdown(f"""
    <div class='lazarus-card animated-card' onclick='this.classList.toggle("pulse")'>
        <div class='metric-header'>
            <span class='metric-icon'>{icon}</span>
            <div class='metric-label'>{title}</div>
        </div>
        <div class='metric-value glow-text'>{value}</div>
        {f"<div class='metric-delta'>{delta}</div>" if delta else ""}
    </div>
    """, unsafe_allow_html=True)


def render_model_status_card(model_key: str, status: str, accuracy: str = "N/A") -> None:
    """Render an interactive model status card."""
    config = MODEL_OPTIONS.get(model_key, {})
    label = config.get("label", model_key)
    trained = config.get("trained", False)
    
    status_color = {
        "Ready": "#34d399",
        "Training": "#fbbf24", 
        "Loading": "#60a5fa",
        "Error": "#f87171"
    }.get(status, "#6b7280")
    
    status_icon = {
        "Ready": "✅",
        "Training": "🔄",
        "Loading": "⏳",
        "Error": "❌"
    }.get(status, "🤖")
    
    st.markdown(f"""
    <div class='model-status-card {"trained-model" if trained else "pretrained-model"}'>
        <div class='model-header'>
            <span class='model-icon'>{status_icon}</span>
            <div class='model-name'>{label}</div>
        </div>
        <div class='model-status' style='color: {status_color}'>{status}</div>
        <div class='model-accuracy'>Accuracy: {accuracy}</div>
        <div class='model-badge'>{"🎯 TRAINED" if trained else "📚 PRETRAINED"}</div>
    </div>
    """, unsafe_allow_html=True)


def render_progress_ring(percentage: float, label: str, color: str = "#4ade80") -> None:
    """Render an animated circular progress ring."""
    circumference = 2 * 3.14159 * 45  # radius = 45
    offset = circumference - (percentage / 100) * circumference
    
    st.markdown(f"""
    <div class='progress-ring-container'>
        <svg class='progress-ring' width='120' height='120'>
            <circle class='progress-ring-bg' cx='60' cy='60' r='45'/>
            <circle class='progress-ring-fill' 
                    cx='60' cy='60' r='45'
                    style='stroke: {color}; stroke-dasharray: {circumference}; 
                           stroke-dashoffset: {offset}; animation: progressAnimation 2s ease-out;'/>
        </svg>
        <div class='progress-ring-text'>
            <div class='progress-percentage'>{percentage:.1f}%</div>
            <div class='progress-label'>{label}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_prediction_confidence_bars(predictions: List[Tuple[str, float]], max_show: int = 5) -> None:
    """Render animated confidence bars for predictions."""
    st.markdown("<div class='confidence-bars-container'>", unsafe_allow_html=True)
    
    for i, (class_name, confidence) in enumerate(predictions[:max_show]):
        bar_color = "#4ade80" if i == 0 else "#60a5fa" if i < 3 else "#94a3b8"
        
        st.markdown(f"""
        <div class='confidence-bar-item'>
            <div class='confidence-label'>{class_name}</div>
            <div class='confidence-bar-bg'>
                <div class='confidence-bar-fill' 
                     style='width: {confidence*100}%; background: {bar_color};
                            animation: barFillAnimation 1.5s ease-out;'></div>
            </div>
            <div class='confidence-value'>{confidence*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)


def render_interactive_heatmap(confusion_matrix: np.ndarray, class_names: List[str]) -> None:
    """Render an interactive confusion matrix heatmap."""
    # Normalize confusion matrix
    cm_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    
    fig = go.Figure(data=go.Heatmap(
        z=cm_normalized,
        x=class_names,
        y=class_names,
        colorscale='Viridis',
        text=confusion_matrix,
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False,
        hovertemplate='<b>Predicted:</b> %{x}<br><b>Actual:</b> %{y}<br><b>Count:</b> %{text}<br><b>Normalized:</b> %{z:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="🎯 Interactive Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        font=dict(family="Inter, sans-serif", size=12),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_realtime_metrics(metrics: Dict[str, float]) -> None:
    """Render real-time animated metrics dashboard."""
    cols = st.columns(4)
    
    with cols[0]:
        render_progress_ring(metrics.get("accuracy", 0) * 100, "Accuracy", "#4ade80")
    
    with cols[1]:
        render_progress_ring(metrics.get("f1_score", 0) * 100, "F1 Score", "#60a5fa")
    
    with cols[2]:
        render_progress_ring(metrics.get("precision", 0) * 100, "Precision", "#fbbf24")
    
    with cols[3]:
        render_progress_ring(metrics.get("recall", 0) * 100, "Recall", "#f87171")


def render_section_divider() -> None:
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)


# --------------------------------------------------------------------------------------------------
# Header Controls
# --------------------------------------------------------------------------------------------------


def render_header() -> None:
    """Render beautiful animated header with theme switcher."""
    if not MODEL_OPTIONS:
        st.error("TorchVision must be installed to use Lazarus Console.")
        return
        
    # Enhanced header with animation and theme switcher
    header_col, theme_col = st.columns([4, 1])
    
    with header_col:
        st.markdown("""
        <div class='header-container'>
            <h1 style='font-size: 3.5rem; margin: 0; background: linear-gradient(45deg, #4ade80, #60a5fa); 
                       -webkit-background-clip: text; -webkit-text-fill-color: transparent; animation: fadeInUp 1s ease-out;'>
                🌱 Lazarus Console
            </h1>
            <p style='font-size: 1.2rem; margin: 0.5rem 0; opacity: 0.8; animation: fadeInUp 1s ease-out 0.2s both;'>
                Immersive Plant Disease Diagnostics • AI-Powered Agriculture
            </p>
            <div style='display: flex; justify-content: center; gap: 2rem; margin: 1rem 0; animation: fadeInUp 1s ease-out 0.4s both;'>
                <span class='feature-badge'>🎯 <strong>Your Trained Models</strong></span>
                <span class='feature-badge'>⚡ <strong>Real-time Inference</strong></span>
                <span class='feature-badge'>🔬 <strong>Explainable AI</strong></span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with theme_col:
        st.markdown("<div style='margin-top: 2rem;'>", unsafe_allow_html=True)
        theme_options = {
            "dark": "🌙 Dark",
            "light": "☀️ Light", 
            "neon": "⚡ Neon"
        }
        new_theme = st.selectbox(
            "Theme",
            options=list(theme_options.keys()),
            format_func=lambda x: theme_options[x],
            index=list(theme_options.keys()).index(st.session_state.theme)
        )
        if new_theme != st.session_state.theme:
            st.session_state.theme = new_theme
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    # Enhanced model and precision selection
    col1, col2, col3, col4, col5 = st.columns([1.2, 1.2, 1.2, 1.1, 1.5])
    
    with col2:
        model_labels = build_model_options()
        selection = st.selectbox(
            "🎛️ Model",
            options=list(MODEL_OPTIONS.keys()),
            format_func=lambda key: model_labels[key],
            index=list(MODEL_OPTIONS.keys()).index(st.session_state.selected_model),
        )
        st.session_state.selected_model = selection

    with col3:
        precision = st.switch_page if False else st.segmented_control(  # type: ignore[attr-defined]
            "Precision",
            options=["AMP", "ONNX"],
            default="AMP" if st.session_state.precision_mode == "AMP" else "ONNX",
            help="AMP uses PyTorch autocast, ONNX leverages ONNX Runtime GPU if available.",
        ) if hasattr(st, "segmented_control") else st.radio(
            "Precision",
            options=["AMP", "ONNX"],
            index=0 if st.session_state.precision_mode == "AMP" else 1,
            horizontal=True,
        )
        st.session_state.precision_mode = precision

    with col4:
        st.session_state.enable_ensemble = st.toggle("🤖 Ensemble", value=st.session_state.enable_ensemble)
        if st.session_state.enable_ensemble:
            weights = {}
            for model_key, cfg in MODEL_OPTIONS.items():
                weights[model_key] = st.slider(
                    f"Weight · {cfg['label']}",
                    min_value=0.0,
                    max_value=3.0,
                    value=float(st.session_state.ensemble_weights.get(model_key, 1.0)),
                    key=f"ensemble_weight_{model_key}",
                )
            total = sum(weights.values()) or 1.0
            st.session_state.ensemble_weights = {k: v / total for k, v in weights.items()}
        else:
            st.session_state.ensemble_weights = DEFAULT_ENSEMBLE_WEIGHTS.copy()

    with col5:
        st.session_state.confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=0.99,
            value=float(st.session_state.confidence_threshold),
            step=0.01,
        )
        st.markdown(
            f"<div class='threshold-highlight'>FN/FP recalculations live at {st.session_state.confidence_threshold:.2f}</div>",
            unsafe_allow_html=True,
        )


# --------------------------------------------------------------------------------------------------
# Home Section
# --------------------------------------------------------------------------------------------------


def render_home_section(metrics_cache: Dict[str, Any]) -> None:
    """Enhanced home section with beautiful interactive metrics."""
    render_section_divider()
    st.markdown("### 🏠 Mission Control Dashboard")
    st.markdown("---")
    
    # Real-time model status cards
    st.markdown("#### 🤖 Model Fleet Status")
    cols = st.columns(len(MODEL_OPTIONS))
    
    for idx, (model_key, config) in enumerate(MODEL_OPTIONS.items()):
        with cols[idx]:
            # Determine model status
            checkpoint_path = get_best_checkpoint_for_model(model_key)
            if checkpoint_path and checkpoint_path.exists():
                status = "Ready"
                accuracy = config.get("actual_accuracy", "N/A")
            else:
                status = "Pretrained Only"
                accuracy = config.get("accuracy_range", "N/A")
            
            render_model_status_card(model_key, status, accuracy)
    
    st.markdown("---")

    latest_run = metrics_cache.get("latest_run") or {}

    def _fmt_metric(value: Optional[float], precision: int = 3) -> str:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return "—"
        return f"{value:.{precision}f}"

    per_class_recall = metrics_cache.get("per_class_recall", np.array([]))
    if isinstance(per_class_recall, list):
        per_class_recall = np.array(per_class_recall)
    best_recall = np.nanmax(per_class_recall) if getattr(per_class_recall, "size", 0) else float("nan")

    latency_value = metrics_cache.get("latency")
    latency_display = "—"
    if isinstance(latency_value, (int, float)) and not math.isnan(latency_value):
        latency_display = format_latency(float(latency_value))

    # Interactive metrics dashboard
    if metrics_cache and metrics_cache.get("total_experiments", 0) > 0:
        st.markdown("#### 📊 Performance Metrics")
        
        # Real-time animated metrics
        if latest_run:
            metrics = {
                "accuracy": latest_run.get("val_accuracy", 0),
                "f1_score": latest_run.get("val_macro_f1", 0),
                "precision": metrics_cache.get("precision", 0),
                "recall": best_recall if not math.isnan(best_recall) else 0
            }
            render_realtime_metrics(metrics)
        
        st.markdown("---")

    # Enhanced metric cards with animations
    metric_cols = st.columns(4)
    with metric_cols[0]:
        render_metric_card("Macro F1", _fmt_metric(metrics_cache.get("macro_f1")), "🎯 Model Performance", "🎯")
    with metric_cols[1]:
        render_metric_card(
            "Critical Recall",
            _fmt_metric(best_recall),
            "🔍 Disease Detection",
            "🔍"
        )
    with metric_cols[2]:
        render_metric_card("Latency", latency_display, "⚡ Speed", "⚡")
    with metric_cols[3]:
        render_metric_card("Model Size", metrics_cache.get("model_size", "—"), "💾 Footprint", "💾")

    if latest_run:
        st.markdown("---")
        summary_parts = []
        macro_val = latest_run.get("val_macro_f1")
        acc_val = latest_run.get("val_accuracy")
        if macro_val is not None:
            summary_parts.append(f"F1 {macro_val:.3f}")
        if acc_val is not None:
            summary_parts.append(f"Acc {acc_val:.3f}")
        tag_line = " · ".join(summary_parts) if summary_parts else "Metrics pending"
        notes = latest_run.get("notes")
        summary_line = (
            f"**Latest Run · {latest_run.get('timestamp', '—')}** — "
            f"{latest_run.get('model_name', '—')} ({latest_run.get('backbone', 'n/a')})"
        )
        st.markdown(summary_line)
        st.caption(tag_line)
        if notes:
            st.caption(f"Notes: {notes}")

    col1, col2 = st.columns([1.4, 1.6])
    with col1:
        st.markdown("### Recent Checkpoints")
        checkpoints = metrics_cache.get("checkpoints", [])
        if checkpoints:
            for ckpt in checkpoints:
                line = f"- **{ckpt['timestamp']}** · {ckpt['model']} → {ckpt['metric']}"
                if ckpt.get("secondary"):
                    line += f" · {ckpt['secondary']}"
                st.markdown(line)
                if ckpt.get("notes"):
                    st.caption(f"{ckpt['notes']}")
        else:
            st.info("No checkpoints recorded yet. Run training to populate this feed.")
    with col2:
        st.markdown("### ⚡ Quick Actions")
        qa_col1, qa_col2 = st.columns(2)
        with qa_col1:
            if st.button("🔬 Start Inference", use_container_width=True):
                st.session_state.current_section = "Inference"
                st.rerun()
        with qa_col2:
            if st.button("🧬 Explainability Studio", use_container_width=True):
                st.session_state.current_section = "Explainability"
                st.rerun()
        
        qa_col3, qa_col4 = st.columns(2)
        with qa_col3:
            if st.button("📊 Compare Models", use_container_width=True):
                st.session_state.current_section = "Model Comparison"
                st.rerun()
        with qa_col4:
            if st.button("🔧 Model Hub", use_container_width=True):
                st.session_state.current_section = "Model Hub"
                st.rerun()
        
        st.caption("Predict now, inspect later. Ensemble toggle supercharges reliability.")


# --------------------------------------------------------------------------------------------------
# Model Hub Section
# --------------------------------------------------------------------------------------------------


def render_model_hub_section() -> None:
    """Display trained models from experiments.csv with download links and metrics."""
    render_section_divider()
    st.subheader("🎯 Model Hub - Trained Checkpoints")
    st.caption("Browse all trained models, view metrics, download artifacts, and load for inference")
    
    if not EXPERIMENTS_INDEX_PATH.exists():
        st.warning("No experiments.csv found. Train your first model to populate the hub!")
        st.markdown("""
        **To train a model:**
        1. Open `notebooks/master_model_trainer.ipynb`
        2. Set `fast_test_mode = True` for a quick run
        3. Execute the notebook cells
        4. Return here to see your trained models
        """)
        return
    
    try:
        df = pd.read_csv(EXPERIMENTS_INDEX_PATH)
    except Exception as e:
        st.error(f"Failed to load experiments.csv: {e}")
        return
    
    if df.empty:
        st.info("No training runs recorded yet. Train your first model!")
        return
    
    # Sort by timestamp (most recent first)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce")
    df = df.sort_values("timestamp_utc", ascending=False)
    
    # Filters
    filter_col1, filter_col2, filter_col3 = st.columns([1, 1, 1])
    with filter_col1:
        model_filter = st.multiselect(
            "Filter by Model",
            options=sorted(df["model_name"].dropna().unique()),
            default=None
        )
    with filter_col2:
        framework_filter = st.multiselect(
            "Filter by Framework",
            options=sorted(df["framework"].dropna().unique()),
            default=None
        )
    with filter_col3:
        min_f1 = st.slider("Minimum F1 Score", 0.0, 1.0, 0.0, 0.05)
    
    # Apply filters
    filtered = df.copy()
    if model_filter:
        filtered = filtered[filtered["model_name"].isin(model_filter)]
    if framework_filter:
        filtered = filtered[filtered["framework"].isin(framework_filter)]
    filtered = filtered[filtered["val_macro_f1"].fillna(0) >= min_f1]
    
    st.markdown(f"**Showing {len(filtered)} of {len(df)} runs**")
    
    # Display runs
    for idx, row in filtered.iterrows():
        with st.expander(
            f"🔹 {row['model_name']} ({row['backbone']}) - F1 {row.get('val_macro_f1', 0):.3f} - {_format_timestamp(row['timestamp_utc'])}",
            expanded=False
        ):
            metrics_col, artifacts_col = st.columns([1.2, 1.0])
            
            with metrics_col:
                st.markdown("#### Metrics")
                met_cols = st.columns(3)
                met_cols[0].metric("Accuracy", f"{row.get('val_accuracy', 0):.3f}")
                met_cols[1].metric("F1 Score", f"{row.get('val_macro_f1', 0):.3f}")
                met_cols[2].metric("Recall", f"{row.get('val_macro_recall', 0):.3f}")
                
                info_cols = st.columns(2)
                info_cols[0].metric("Epochs", int(row.get("epochs_trained", 0)))
                info_cols[1].metric("Batch Size", int(row.get("batch_size", 0)))
                
                st.markdown(f"**Run ID:** `{row['run_id']}`")
                st.markdown(f"**Framework:** {row.get('framework', 'unknown')}")
                st.markdown(f"**Input Size:** {row.get('input_size', 0)}×{row.get('input_size', 0)}")
                st.markdown(f"**Parameters:** {int(row.get('params_count', 0)):,}")
                
                notes = row.get("notes")
                if notes and str(notes).strip():
                    st.markdown(f"**Notes:** {notes}")
            
            with artifacts_col:
                st.markdown("#### Artifacts")
                
                # PyTorch checkpoint
                checkpoint_path = row.get("best_checkpoint_path")
                if checkpoint_path and pd.notna(checkpoint_path):
                    full_path = PROJECT_ROOT / checkpoint_path
                    if full_path.exists():
                        st.markdown(f"✅ PyTorch: `{checkpoint_path}`")
                        if st.button(f"Load for Inference", key=f"load_{idx}"):
                            st.session_state.selected_checkpoint = str(full_path)
                            st.success(f"✓ Loaded {row['model_name']} for inference")
                    else:
                        st.markdown(f"⚠️ PyTorch: Not found")
                
                # ONNX export
                onnx_path = row.get("onnx_path")
                if onnx_path and pd.notna(onnx_path) and str(onnx_path).strip():
                    full_onnx = PROJECT_ROOT / onnx_path
                    if full_onnx.exists():
                        st.markdown(f"✅ ONNX: `{onnx_path}`")
                    else:
                        st.markdown(f"⚠️ ONNX: Export failed")
                else:
                    st.markdown("➖ ONNX: Not exported")
                
                # TFLite export
                tflite_path = row.get("tflite_path")
                if tflite_path and pd.notna(tflite_path) and str(tflite_path).strip():
                    full_tflite = PROJECT_ROOT / tflite_path
                    if full_tflite.exists():
                        st.markdown(f"✅ TFLite: `{tflite_path}`")
                    else:
                        st.markdown(f"⚠️ TFLite: Export failed")
                else:
                    st.markdown("➖ TFLite: Not exported")
                
                # Grad-CAM gallery
                gradcam_folder = row.get("gradcam_folder")
                if gradcam_folder and pd.notna(gradcam_folder):
                    full_gradcam = PROJECT_ROOT / gradcam_folder
                    if full_gradcam.exists():
                        gradcam_images = list(full_gradcam.glob("*.png"))
                        st.markdown(f"✅ Grad-CAM: {len(gradcam_images)} images")
                        if st.button(f"View Gallery", key=f"gradcam_{idx}"):
                            st.session_state.gradcam_gallery = gradcam_images[:6]
                    else:
                        st.markdown("⚠️ Grad-CAM: Not found")
    
    # Show Grad-CAM gallery if selected
    if "gradcam_gallery" in st.session_state and st.session_state.gradcam_gallery:
        st.markdown("---")
        st.markdown("### Grad-CAM Gallery")
        cols = st.columns(3)
        for idx, img_path in enumerate(st.session_state.gradcam_gallery):
            with cols[idx % 3]:
                st.image(str(img_path), caption=img_path.name, use_container_width=True)
        if st.button("Close Gallery"):
            del st.session_state.gradcam_gallery


# --------------------------------------------------------------------------------------------------
# Inference Section
# --------------------------------------------------------------------------------------------------


def render_inference_section() -> None:
    render_section_divider()
    st.subheader("Inference Laboratory")
    st.caption("Upload batches, toggle precision, and monitor FN/FP trade-offs in real-time.")

    upload_col, settings_col = st.columns([1.5, 1.0])
    with upload_col:
        uploaded = st.file_uploader(
            "Drop plant leaf images",
            accept_multiple_files=True,
            type=["png", "jpg", "jpeg"],
        )
    with settings_col:
        include_gallery = st.checkbox("Show gallery", value=True)
        enable_csv = st.checkbox("Enable CSV export", value=True)

    if not uploaded:
        st.info("Upload at least one image to run inference.")
        return

    images = [Image.open(file).convert("RGB") for file in uploaded]
    backend = "PyTorch" if st.session_state.precision_mode == "AMP" else "ONNX"
    use_amp = st.session_state.precision_mode == "AMP"
    model_keys = list(MODEL_OPTIONS.keys())
    results: List[InferenceResult] = []

    for model_key in model_keys:
        try:
            warmup_once(model_key, backend)
            result = run_single_backend_inference(images, model_key, backend, use_amp)
            results.append(result)
        except Exception as exc:
            toast_warning(f"{MODEL_OPTIONS[model_key]['label']} failed: {exc}")

    if not results:
        st.error("No inference results available.")
        return

    weight_template = st.session_state.ensemble_weights or DEFAULT_ENSEMBLE_WEIGHTS
    ensemble_weights = {res.model_key: float(weight_template.get(res.model_key, 1.0)) for res in results}
    blended = blend_logits(results, ensemble_weights) if len(results) > 1 else results[0]

    class_names = load_class_names()
    probs_combined = blended.probs
    combined_top_indices = np.argmax(probs_combined, axis=1)
    combined_confidences = np.max(probs_combined, axis=1)
    combined_predictions = [class_names[idx] for idx in combined_top_indices]

    # Assemble batch summary with per-model contributions
    summary_rows: List[Dict[str, Any]] = []
    for row_idx, file_obj in enumerate(uploaded):
        row: Dict[str, Any] = {
            "Image": file_obj.name,
            "Combined Prediction": combined_predictions[row_idx],
            "Combined Confidence": combined_confidences[row_idx],
        }
        for res in results:
            label = MODEL_OPTIONS[res.model_key]["label"]
            per_probs = res.probs[row_idx]
            per_idx = int(np.argmax(per_probs))
            row[f"{label} Prediction"] = class_names[per_idx]
            row[f"{label} Confidence"] = float(per_probs[per_idx])
        summary_rows.append(row)

    df_results = pd.DataFrame(summary_rows)
    st.markdown("#### Combined Batch Summary")
    st.dataframe(df_results, use_container_width=True)

    avg_latency = float(np.mean([res.latency_ms for res in results]))
    st.metric("Avg model latency", format_latency(avg_latency))

    if enable_csv:
        csv_bytes = df_results.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Export CSV",
            data=csv_bytes,
            file_name="lazarus_inference_results.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.markdown("#### Detailed Per-image Analysis")
    threshold = st.session_state.confidence_threshold
    for idx, (img, file_obj) in enumerate(zip(images, uploaded)):
        exceeds_threshold = combined_confidences[idx] >= threshold
        header = (
            f"{file_obj.name} · {combined_predictions[idx]} ({combined_confidences[idx]:.1%})"
            f"{' ✅' if exceeds_threshold else ' ⚠️'}"
        )
        with st.expander(header, expanded=len(images) == 1):
            st.image(img, caption="Uploaded sample", use_column_width=True)

            sorted_indices = np.argsort(probs_combined[idx])[::-1]
            top_indices = sorted_indices[: min(10, len(class_names))]
            chart_df = pd.DataFrame(
                {
                    "Disease": [class_names[i] for i in top_indices],
                    "Probability": probs_combined[idx][top_indices],
                }
            )
            fig = px.bar(
                chart_df,
                x="Disease",
                y="Probability",
                text=chart_df["Probability"].map(lambda p: f"{p:.1%}"),
                title="Combined probability mass",
            )
            fig.update_layout(yaxis=dict(range=[0, 1]))
            st.plotly_chart(fig, use_container_width=True)

            model_rows = []
            for res in results:
                label = MODEL_OPTIONS[res.model_key]["label"]
                per_probs = res.probs[idx]
                per_idx = int(np.argmax(per_probs))
                model_rows.append(
                    {
                        "Model": label,
                        "Prediction": class_names[per_idx],
                        "Confidence": per_probs[per_idx],
                        "Latency (ms)": res.latency_ms / len(images),
                    }
                )
            model_df = pd.DataFrame(model_rows)
            model_df["Confidence"] = model_df["Confidence"].map(lambda v: f"{v:.1%}")
            model_df["Latency (ms)"] = model_df["Latency (ms)"].map(lambda v: f"{v:.1f}")
            st.dataframe(model_df, use_container_width=True)

    if include_gallery:
        st.markdown("#### Gallery")
        gallery_cols = st.columns(3)
        for idx, img in enumerate(images):
            with gallery_cols[idx % 3]:
                caption = f"{combined_predictions[idx]} ({combined_confidences[idx]:.1%})"
                st.image(img, caption=caption, use_column_width=True)

    st.session_state.inference_history.append(
        {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "backend": backend,
            "models": model_keys,
            "count": len(images),
        }
    )


# --------------------------------------------------------------------------------------------------
# Explainability Section
# --------------------------------------------------------------------------------------------------


def render_explainability_section() -> None:
    render_section_divider()
    st.subheader("Explainability Studio")

    if not st.session_state.inference_history:
        st.info("Run an inference batch to unlock Grad-CAM explainability.")
        return

    explainer_col, gallery_col = st.columns([1.2, 1.4])

    with explainer_col:
        st.markdown("#### Select Reference Image")
        manifest = build_dataset_manifest(sample_per_class=4)
        sample_images = list(manifest.frame["path"])
        sample_images = [str(path) for path in sample_images[:12]]
        if not sample_images:
            st.warning("No sample images available in data directory.")
            return
        selected_path = cast(str, st.selectbox("Sample image", sample_images))
        target_image = Image.open(selected_path).convert("RGB")
        st.image(target_image, caption=Path(selected_path).name, use_column_width=True)

        blend = st.slider("Grad-CAM Blend", 0.0, 1.0, 0.45, 0.05)
        topk = st.slider("Top-K Overlays", 1, 5, 3)
        if st.button("Generate Grad-CAM", use_container_width=True):
            with st.spinner("Backpropagating explanations..."):
                model = load_torch_model(st.session_state.selected_model)
                if model is None:
                    st.error("PyTorch model unavailable for Grad-CAM")
                    return
                transform = get_preprocess_transform(MODEL_OPTIONS[st.session_state.selected_model]["input_size"])
                tensor = transform(target_image).unsqueeze(0).to(get_device())
                logits = model(tensor)
                assert torch is not None
                probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
                top_indices = np.argsort(probs)[::-1][:topk]
                target_class = top_indices[0]
                target_layer_name = "features" if "efficientnet" in st.session_state.selected_model else "layer4"
                cam = generate_gradcam(model, tensor, target_layer_name, target_class)
                heatmap = Image.fromarray(np.uint8(cam * 255)).resize(target_image.size)
                heatmap = heatmap.convert("RGBA")
                base = target_image.convert("RGBA")
                overlay = Image.blend(base, heatmap, alpha=blend)
                st.session_state.explainability_cache = {
                    "overlay": overlay,
                    "top_classes": [(load_class_names()[idx], probs[idx]) for idx in top_indices],
                }
                toast_success("Grad-CAM generated")

    with gallery_col:
        cache = st.session_state.explainability_cache
        if cache:
            st.markdown("#### Overlay Preview")
            st.image(cache["overlay"], caption="Grad-CAM Overlay", use_column_width=True)
            st.markdown("#### Top Classes")
            for cls, prob in cache["top_classes"]:
                st.markdown(f"- **{cls}** · {prob:.2%}")
        else:
            st.info("Generate Grad-CAM to view overlay and top classes.")


# --------------------------------------------------------------------------------------------------
# Compare Section
# --------------------------------------------------------------------------------------------------


def render_compare_section() -> None:
    render_section_divider()
    st.subheader("Model & Ensemble Comparison")
    backend = "PyTorch" if st.session_state.precision_mode == "AMP" else "ONNX"
    threshold = st.session_state.confidence_threshold

    comparison_data = {}
    latency_estimates = {}
    for model_key in MODEL_OPTIONS.keys():
        start = time.perf_counter()
        metrics = cached_model_metrics(model_key, backend, threshold)
        latency_estimates[model_key] = (time.perf_counter() - start) * 1000
        metrics["latency"] = latency_estimates[model_key]
        metrics["model_size"] = "~20 MB" if "mobilenet" in model_key else "~45 MB"
        comparison_data[model_key] = metrics

    if st.session_state.enable_ensemble:
        logits = []
        weights = st.session_state.ensemble_weights
        for model_key in MODEL_OPTIONS.keys():
            logits.append(
                cached_model_metrics(model_key, backend, threshold)["confusion_matrix"]
            )
        ensemble_metrics = cached_model_metrics(st.session_state.selected_model, backend, threshold)
        ensemble_metrics["macro_f1"] = np.mean([comparison_data[m]["macro_f1"] for m in MODEL_OPTIONS])
        ensemble_metrics["per_class_recall"] = np.mean(
            [comparison_data[m]["per_class_recall"] for m in MODEL_OPTIONS], axis=0
        )
        ensemble_metrics["latency"] = max(latency_estimates.values()) * 1.25
        ensemble_metrics["model_size"] = "Aggregated"
        comparison_data["ensemble"] = ensemble_metrics

    tabs = st.tabs([MODEL_OPTIONS[k]["label"] for k in MODEL_OPTIONS] + (["Ensemble"] if st.session_state.enable_ensemble else []))
    for tab, model_key in zip(tabs, comparison_data.keys()):
        with tab:
            metrics = comparison_data[model_key]
            met_col1, met_col2, met_col3 = st.columns(3)
            met_col1.metric("Macro F1", f"{metrics['macro_f1']:.3f}")
            met_col2.metric("Latency", f"{metrics['latency']:.1f} ms")
            met_col3.metric("ECE", f"{metrics['ece']:.3f}")

            fig_conf = px.imshow(
                metrics["confusion_matrix"],
                text_auto=True,
                color_continuous_scale="Blues",
                labels=dict(x="Predicted", y="Actual", color="Count"),
            )
            st.plotly_chart(fig_conf, use_container_width=True)

            fig_calib = go.Figure()
            fig_calib.add_trace(
                go.Scatter(x=metrics["avg_confidence"], y=metrics["avg_accuracy"], mode="lines+markers", name="Observed")
            )
            fig_calib.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect"))
            fig_calib.update_layout(title="Reliability Diagram", xaxis_title="Confidence", yaxis_title="Accuracy")
            st.plotly_chart(fig_calib, use_container_width=True)

            recalls = metrics["per_class_recall"]
            class_names = load_class_names()
            fig_recall = go.Figure(
                go.Bar(x=class_names, y=recalls, marker_color="#7dd87d" if st.session_state.theme == "dark" else "#2b6cb0")
            )
            fig_recall.update_layout(title="Per-class Recall", xaxis_title="Class", yaxis_title="Recall")
            st.plotly_chart(fig_recall, use_container_width=True)


# --------------------------------------------------------------------------------------------------
# Footer
# --------------------------------------------------------------------------------------------------


def render_footer() -> None:
    render_section_divider()
    st.markdown(
        """
        <div style="text-align:center; opacity:0.8; padding:1.5rem 0;">
            <h4 style="margin-bottom:0.3rem;">🚀 Ready for Deployment</h4>
            <p style="margin:0;">Latency, calibration, threshold sweeps, and Grad-CAM insights converge here. Lazarus Console keeps agronomists in the loop—and the crops thriving.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------------------------------
# Main App
# --------------------------------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="Lazarus Console", page_icon="🌱", layout="wide")
    ensure_session_state()
    inject_theme(st.session_state.theme)

    st.sidebar.title("🧭 Navigation")
    section = st.sidebar.radio(
        "Go to section:",
        ["Home", "Model Hub", "Inference Lab", "Explainability", "Model Comparison"],
        index=0
    )
    
    st.sidebar.markdown("---")
    st.sidebar.title("Notebook-style Insights")
    st.sidebar.write("""
    - 🛰️ Ensemble toggles enable consensus for high-stakes deployments.
    - 🔬 Grad-CAM overlays decode model attention without leaving the console.
    - 🧪 Calibration and ECE charts drive threshold negotiation with agronomists.
    - ⚙️ Warmups ensure sub-500 ms latency for batch=1 on compact backbones.
    """)
    st.sidebar.write("Inference History")
    if st.session_state.inference_history:
        for entry in reversed(st.session_state.inference_history[-6:]):
            models_run = entry.get("models") or [entry.get("model", "-")]
            model_label = ", ".join(models_run)
            st.sidebar.write(f"{entry['timestamp']} · {model_label} · {entry['backend']} · {entry['count']} imgs")
    else:
        st.sidebar.info("No inferences yet.")

    render_header()
    if not MODEL_OPTIONS:
        st.stop()

    # Render selected section
    if section == "Home":
        metrics_cache = build_home_metrics()
        render_home_section(metrics_cache)
    elif section == "Model Hub":
        render_model_hub_section()
    elif section == "Inference Lab":
        render_inference_section()
    elif section == "Explainability":
        render_explainability_section()
    elif section == "Model Comparison":
        render_compare_section()
    
    render_footer()


if __name__ == "__main__":
    main()
