"""
Training Monitor Route - Lazarus Console
Live training monitoring with curves, checkpoints, and controls
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime
from components.state_manager import get_state, set_state, add_decision_log_entry

def render_training_monitor():
    """Render training monitoring interface"""
    
    st.markdown("## 🏋️ Training Monitor")
    st.markdown("*Live training monitoring and checkpoint management*")
    
    # Training controls
    render_training_controls()
    
    # Training metrics
    col1, col2 = st.columns([2, 1])
    
    with col1:
        render_training_curves()
        render_learning_rate_schedule()
    
    with col2:
        render_checkpoint_timeline()
        render_training_logs()

def render_training_controls():
    """Render training control panel"""
    
    training_status = get_state('training_run_status', 'idle')
    
    st.markdown("### Training Controls")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if training_status == 'idle':
            if st.button("▶️ Start Training", type="primary"):
                set_state('training_run_status', 'running')
                add_decision_log_entry("training_started", {"timestamp": datetime.now().isoformat()})
                st.success("Training started!")
                st.rerun()
    
    with col2:
        if training_status == 'running':
            if st.button("⏸️ Pause"):
                set_state('training_run_status', 'paused')
                st.info("Training paused")
                st.rerun()
    
    with col3:
        if training_status in ['running', 'paused']:
            if st.button("⏹️ Stop"):
                set_state('training_run_status', 'stopped')
                st.warning("Training stopped")
                st.rerun()
    
    with col4:
        if training_status == 'paused':
            if st.button("▶️ Resume", type="primary"):
                set_state('training_run_status', 'running')
                st.success("Training resumed")
                st.rerun()
    
    with col5:
        st.metric("Status", training_status.upper())

def render_training_curves():
    """Render live training curves"""
    
    st.markdown("### Training Metrics")
    
    # Simulate training data
    training_metrics = get_state('training_metrics', generate_mock_training_data())
    
    if not training_metrics:
        st.info("No training data available")
        return
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Loss', 'Accuracy', 'F1 Score', 'Learning Rate'],
        vertical_spacing=0.1
    )
    
    epochs = list(range(1, len(training_metrics['train_loss']) + 1))
    
    # Loss curves
    fig.add_trace(
        go.Scatter(x=epochs, y=training_metrics['train_loss'], name='Train Loss', line=dict(color='red')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=epochs, y=training_metrics['val_loss'], name='Val Loss', line=dict(color='orange')),
        row=1, col=1
    )
    
    # Accuracy curves
    fig.add_trace(
        go.Scatter(x=epochs, y=training_metrics['train_acc'], name='Train Acc', line=dict(color='blue')),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=epochs, y=training_metrics['val_acc'], name='Val Acc', line=dict(color='lightblue')),
        row=1, col=2
    )
    
    # F1 Score curves
    fig.add_trace(
        go.Scatter(x=epochs, y=training_metrics['train_f1'], name='Train F1', line=dict(color='green')),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=epochs, y=training_metrics['val_f1'], name='Val F1', line=dict(color='lightgreen')),
        row=2, col=1
    )
    
    # Learning rate
    lr_values = [0.001 * (0.95 ** (epoch // 5)) for epoch in range(len(epochs))]
    fig.add_trace(
        go.Scatter(x=epochs, y=lr_values, name='LR', line=dict(color='purple')),
        row=2, col=2
    )
    
    fig.update_layout(
        height=600,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_learning_rate_schedule():
    """Render learning rate schedule"""
    
    st.markdown("### Current Metrics")
    
    training_metrics = get_state('training_metrics', generate_mock_training_data())
    
    if training_metrics:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current Loss", f"{training_metrics['train_loss'][-1]:.4f}")
        
        with col2:
            st.metric("Current Accuracy", f"{training_metrics['train_acc'][-1]:.3f}")
        
        with col3:
            st.metric("Current F1", f"{training_metrics['train_f1'][-1]:.3f}")
        
        with col4:
            st.metric("Epoch", len(training_metrics['train_loss']))

def render_checkpoint_timeline():
    """Render checkpoint timeline"""
    
    st.markdown("### Checkpoints")
    
    checkpoints = get_state('checkpoint_timeline', generate_mock_checkpoints())
    
    for i, checkpoint in enumerate(checkpoints):
        with st.container():
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.text(f"Checkpoint {i+1}")
                st.text(f"Epoch {checkpoint['epoch']}")
            
            with col2:
                st.metric("F1", f"{checkpoint['f1_score']:.3f}")
            
            with col3:
                if st.button("Load", key=f"load_cp_{i}"):
                    st.success(f"Loaded checkpoint {i+1}")
            
            st.divider()

def render_training_logs():
    """Render training logs"""
    
    st.markdown("### Training Logs")
    
    # Mock log entries
    logs = [
        "[INFO] Training started with 32 batch size",
        "[INFO] Epoch 1/50 - Loss: 0.8234, Acc: 0.7123",
        "[INFO] Epoch 2/50 - Loss: 0.7456, Acc: 0.7589",
        "[INFO] Checkpoint saved at epoch 5",
        "[WARNING] Learning rate reduced to 0.0005",
        "[INFO] Epoch 10/50 - Loss: 0.4521, Acc: 0.8456"
    ]
    
    log_text = "\n".join(logs[-10:])  # Show last 10 logs
    st.text_area("Recent Logs", value=log_text, height=200, disabled=True)

def generate_mock_training_data():
    """Generate mock training data for demonstration"""
    epochs = 20
    
    # Simulate realistic training curves
    train_loss = [1.5 * np.exp(-0.1 * i) + 0.1 + np.random.normal(0, 0.02) for i in range(epochs)]
    val_loss = [1.6 * np.exp(-0.08 * i) + 0.15 + np.random.normal(0, 0.03) for i in range(epochs)]
    
    train_acc = [0.6 + 0.35 * (1 - np.exp(-0.12 * i)) + np.random.normal(0, 0.01) for i in range(epochs)]
    val_acc = [0.55 + 0.35 * (1 - np.exp(-0.1 * i)) + np.random.normal(0, 0.015) for i in range(epochs)]
    
    train_f1 = [0.5 + 0.4 * (1 - np.exp(-0.1 * i)) + np.random.normal(0, 0.01) for i in range(epochs)]
    val_f1 = [0.45 + 0.4 * (1 - np.exp(-0.08 * i)) + np.random.normal(0, 0.015) for i in range(epochs)]
    
    return {
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'train_f1': train_f1,
        'val_f1': val_f1
    }

def generate_mock_checkpoints():
    """Generate mock checkpoint data"""
    return [
        {'epoch': 5, 'f1_score': 0.756, 'timestamp': '2024-01-01 10:15:30'},
        {'epoch': 10, 'f1_score': 0.823, 'timestamp': '2024-01-01 10:25:45'},
        {'epoch': 15, 'f1_score': 0.867, 'timestamp': '2024-01-01 10:35:12'},
        {'epoch': 20, 'f1_score': 0.891, 'timestamp': '2024-01-01 10:44:28'}
    ]