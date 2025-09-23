"""
CAPSTONE-LAZARUS: Quantum Layer Module
=====================================
Optional quantum-classical hybrid layers using PennyLane.
EXPERIMENTAL - Default OFF for production training.
"""

import torch
import torch.nn as nn
from typing import Optional
import logging

logger = logging.getLogger(__name__)

try:
    import pennylane as qml
    from pennylane import numpy as np
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False
    logger.warning("PennyLane not available. Quantum layers disabled.")


class QuantumLayer(nn.Module):
    """
    Quantum Neural Network Layer using PennyLane.
    
    This is an EXPERIMENTAL feature for research purposes.
    Default behavior is to use classical layers only.
    """
    
    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 3,
        embedding_dim: int = 64,
        device: str = "default.qubit"
    ):
        super().__init__()
        
        if not PENNYLANE_AVAILABLE:
            raise ImportError(
                "PennyLane is required for quantum layers. "
                "Install with: pip install pennylane"
            )
        
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.embedding_dim = embedding_dim
        
        # Create quantum device
        self.dev = qml.device(device, wires=n_qubits)
        
        # Quantum circuit
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            # Angle embedding
            qml.templates.AngleEmbedding(inputs[:self.n_qubits], wires=range(n_qubits))
            
            # Variational layers
            qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            
            # Measurements
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        
        # Convert to PyTorch layer
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.quantum_layer = qml.qnn.TorchLayer(circuit, weight_shapes)
        
        # Input projection if needed
        if embedding_dim > n_qubits:
            self.input_proj = nn.Linear(embedding_dim, n_qubits)
        else:
            self.input_proj = None
            
        logger.info(
            f"Quantum layer created: {n_qubits} qubits, {n_layers} layers, "
            f"device: {device}"
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        
        # Project input to qubit dimension if needed
        if self.input_proj is not None:
            x = self.input_proj(x)
        else:
            # Take first n_qubits features
            x = x[:, :self.n_qubits]
        
        # Process each sample in the batch
        outputs = []
        for i in range(batch_size):
            sample = x[i]
            output = self.quantum_layer(sample)
            outputs.append(output)
        
        return torch.stack(outputs)


class HybridQuantumClassifier(nn.Module):
    """
    Hybrid quantum-classical classifier.
    
    Combines classical feature extraction with quantum processing
    for the final classification layers.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        n_qubits: int = 4,
        n_layers: int = 3,
        classical_hidden_dim: int = 128
    ):
        super().__init__()
        
        # Classical preprocessing
        self.classical_layers = nn.Sequential(
            nn.Linear(input_dim, classical_hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(classical_hidden_dim),
            nn.Dropout(0.2)
        )
        
        # Quantum processing
        self.quantum_layer = QuantumLayer(
            n_qubits=n_qubits,
            n_layers=n_layers,
            embedding_dim=classical_hidden_dim
        )
        
        # Classical output
        self.output_layer = nn.Sequential(
            nn.Linear(n_qubits, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Classical feature processing
        x = self.classical_layers(x)
        
        # Quantum processing
        x = self.quantum_layer(x)
        
        # Final classification
        x = self.output_layer(x)
        
        return x


def create_quantum_model(
    backbone_features: int,
    num_classes: int,
    quantum_config: dict
) -> HybridQuantumClassifier:
    """
    Create a hybrid quantum-classical model.
    
    Args:
        backbone_features: Number of features from backbone
        num_classes: Number of output classes
        quantum_config: Quantum layer configuration
        
    Returns:
        Hybrid quantum-classical model
    """
    
    if not PENNYLANE_AVAILABLE:
        raise ImportError(
            "PennyLane is required for quantum models. "
            "Install with: pip install pennylane"
        )
    
    model = HybridQuantumClassifier(
        input_dim=backbone_features,
        num_classes=num_classes,
        n_qubits=quantum_config.get('n_qubits', 4),
        n_layers=quantum_config.get('n_layers', 3),
        classical_hidden_dim=quantum_config.get('embedding_dim', 128)
    )
    
    logger.info(
        f"Hybrid quantum model created: "
        f"{quantum_config.get('n_qubits', 4)} qubits, "
        f"{quantum_config.get('n_layers', 3)} layers"
    )
    
    return model


# Quantum circuit visualization (for debugging)
def visualize_quantum_circuit(n_qubits: int = 4, n_layers: int = 3):
    """Visualize the quantum circuit structure."""
    
    if not PENNYLANE_AVAILABLE:
        print("PennyLane not available for visualization")
        return
    
    dev = qml.device("default.qubit", wires=n_qubits)
    
    @qml.qnode(dev)
    def circuit(weights):
        qml.templates.AngleEmbedding([0.1] * n_qubits, wires=range(n_qubits))
        qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    
    # Dummy weights
    weights = np.random.random((n_layers, n_qubits, 3))
    
    print("Quantum Circuit Structure:")
    print(qml.draw(circuit)(weights))


# Performance warning
def quantum_performance_warning():
    """Warning about quantum simulation performance."""
    logger.warning(
        "QUANTUM LAYER PERFORMANCE WARNING:\n"
        "Quantum simulation is computationally expensive, especially on CPU.\n"
        "Use quantum layers only for research/experimentation.\n"
        "For production training, keep use_quantum=False in config.yaml"
    )


if __name__ == "__main__":
    # Test quantum layer if PennyLane is available
    if PENNYLANE_AVAILABLE:
        print("Testing Quantum Layer...")
        quantum_performance_warning()
        
        # Create test layer
        ql = QuantumLayer(n_qubits=4, n_layers=2, embedding_dim=8)
        
        # Test forward pass
        test_input = torch.randn(2, 8)
        output = ql(test_input)
        print(f"Input shape: {test_input.shape}")
        print(f"Output shape: {output.shape}")
        
        # Visualize circuit
        visualize_quantum_circuit()
    else:
        print("PennyLane not available. Quantum layers disabled.")