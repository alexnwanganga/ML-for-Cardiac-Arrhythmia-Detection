from __future__ import annotations

import math
from functools import partial
from pathlib import Path
from typing import Literal

import pennylane as qml
import torch
from torch import nn

from .models import ECGEncoder


ModelName = Literal["linear", "matched-mlp", "hybrid-vqc", "hybrid-qcnn"]


def vqc_parameter_count(n_qubits: int, quantum_depth: int) -> int:
    return quantum_depth * n_qubits * 3


def qcnn_parameter_count(n_qubits: int, quantum_depth: int) -> int:
    levels = int(math.log2(n_qubits))
    return levels * quantum_depth * n_qubits * 3 + (n_qubits - 1) * 2


def _validate_quantum_shape(n_qubits: int, quantum_depth: int) -> None:
    if n_qubits < 2 or n_qubits & (n_qubits - 1):
        raise ValueError("n_qubits must be a power of two and at least 2")
    if quantum_depth < 1:
        raise ValueError("quantum_depth must be positive")


class VariationalQuantumFeatures(nn.Module):
    def __init__(
        self,
        n_qubits: int,
        quantum_depth: int,
        device_name: str = "default.qubit",
        *,
        embedding: Literal["angle", "reupload"] = "angle",
        shots: int | None = None,
        noise_probability: float = 0.0,
    ) -> None:
        super().__init__()
        _validate_quantum_shape(n_qubits, quantum_depth)
        self.n_qubits = n_qubits
        if not 0.0 <= noise_probability < 1.0:
            raise ValueError("noise_probability must be in [0, 1)")
        if noise_probability > 0 and device_name == "default.qubit":
            device_name = "default.mixed"
        device = qml.device(device_name, wires=n_qubits, shots=shots)
        diff_method = "parameter-shift" if shots is not None or noise_probability > 0 else "backprop"

        @qml.qnode(device, interface="torch", diff_method=diff_method)
        def circuit(inputs: torch.Tensor, weights: torch.Tensor):
            if embedding == "reupload":
                for layer in range(quantum_depth):
                    qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
                    qml.StronglyEntanglingLayers(weights[layer : layer + 1], wires=range(n_qubits))
            else:
                qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
                qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            if noise_probability > 0:
                for wire in range(n_qubits):
                    qml.DepolarizingChannel(noise_probability, wires=wire)
            return [qml.expval(qml.PauliZ(wire)) for wire in range(n_qubits)]

        weight_shapes = {"weights": (quantum_depth, n_qubits, 3)}
        self.layer = qml.qnn.TorchLayer(
            circuit,
            weight_shapes,
            init_method={"weights": partial(nn.init.normal_, mean=0.0, std=0.1)},
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.layer(inputs)
        return output.to(dtype=inputs.dtype)


class QCNNFeatures(nn.Module):
    """Hierarchical convolution/pooling circuit with multiclass-ready outputs."""

    def __init__(
        self,
        n_qubits: int,
        quantum_depth: int,
        device_name: str = "default.qubit",
        *,
        embedding: Literal["angle", "reupload"] = "angle",
        shots: int | None = None,
        noise_probability: float = 0.0,
    ) -> None:
        super().__init__()
        _validate_quantum_shape(n_qubits, quantum_depth)
        self.n_qubits = n_qubits
        levels = int(math.log2(n_qubits))
        if not 0.0 <= noise_probability < 1.0:
            raise ValueError("noise_probability must be in [0, 1)")
        if noise_probability > 0 and device_name == "default.qubit":
            device_name = "default.mixed"
        device = qml.device(device_name, wires=n_qubits, shots=shots)
        diff_method = "parameter-shift" if shots is not None or noise_probability > 0 else "backprop"

        @qml.qnode(device, interface="torch", diff_method=diff_method)
        def circuit(
            inputs: torch.Tensor,
            convolution_weights: torch.Tensor,
            pooling_weights: torch.Tensor,
        ):
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
            active_wires = list(range(n_qubits))
            pool_index = 0
            for level in range(levels):
                if embedding == "reupload" and level > 0:
                    qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
                for depth in range(quantum_depth):
                    for wire in active_wires:
                        qml.Rot(*convolution_weights[level, depth, wire], wires=wire)
                    for left, right in zip(active_wires, active_wires[1:] + active_wires[:1]):
                        qml.CNOT(wires=[left, right])
                sources = active_wires[::2]
                sinks = active_wires[1::2]
                for source, sink in zip(sources, sinks):
                    qml.CRY(pooling_weights[pool_index, 0], wires=[source, sink])
                    qml.CRX(pooling_weights[pool_index, 1], wires=[source, sink])
                    pool_index += 1
                active_wires = sinks
            if noise_probability > 0:
                for wire in range(n_qubits):
                    qml.DepolarizingChannel(noise_probability, wires=wire)
            return [qml.expval(qml.PauliZ(wire)) for wire in range(n_qubits)]

        weight_shapes = {
            "convolution_weights": (levels, quantum_depth, n_qubits, 3),
            "pooling_weights": (n_qubits - 1, 2),
        }
        init_method = {
            "convolution_weights": partial(nn.init.normal_, mean=0.0, std=0.1),
            "pooling_weights": partial(nn.init.normal_, mean=0.0, std=0.1),
        }
        self.layer = qml.qnn.TorchLayer(circuit, weight_shapes, init_method=init_method)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.layer(inputs)
        return output.to(dtype=inputs.dtype)


class SharedEncoderClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        *,
        encoder_dim: int,
        n_qubits: int,
        dropout: float,
        feature_layer: nn.Module,
    ) -> None:
        super().__init__()
        self.encoder = ECGEncoder(in_channels=12, latent_dim=encoder_dim, dropout=dropout)
        self.bottleneck = nn.Linear(encoder_dim, n_qubits)
        self.feature_layer = feature_layer
        self.classifier = nn.Linear(n_qubits, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        embedding = self.encoder(inputs)
        angles = math.pi * torch.tanh(self.bottleneck(embedding))
        features = self.feature_layer(angles)
        return self.classifier(features)


class LinearFeatures(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs


class ParameterMatchedMLPFeatures(nn.Module):
    def __init__(self, n_features: int, target_parameters: int) -> None:
        super().__init__()
        candidates = range(1, 129)
        hidden = min(
            candidates,
            key=lambda width: abs((2 * n_features * width + width + n_features) - target_parameters),
        )
        self.target_parameters = target_parameters
        self.hidden_width = hidden
        self.network = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_features),
            nn.Tanh(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def build_comparison_model(
    model_name: ModelName,
    num_classes: int,
    *,
    encoder_dim: int = 32,
    n_qubits: int = 4,
    quantum_depth: int = 2,
    dropout: float = 0.25,
    quantum_device: str = "default.qubit",
    matched_to: Literal["vqc", "qcnn"] = "vqc",
    embedding: Literal["angle", "reupload"] = "angle",
    shots: int | None = None,
    noise_probability: float = 0.0,
) -> SharedEncoderClassifier:
    _validate_quantum_shape(n_qubits, quantum_depth)
    if model_name == "linear":
        features: nn.Module = LinearFeatures()
    elif model_name == "matched-mlp":
        target = (
            vqc_parameter_count(n_qubits, quantum_depth)
            if matched_to == "vqc"
            else qcnn_parameter_count(n_qubits, quantum_depth)
        )
        features = ParameterMatchedMLPFeatures(n_qubits, target)
    elif model_name == "hybrid-vqc":
        features = VariationalQuantumFeatures(
            n_qubits,
            quantum_depth,
            quantum_device,
            embedding=embedding,
            shots=shots,
            noise_probability=noise_probability,
        )
    elif model_name == "hybrid-qcnn":
        features = QCNNFeatures(
            n_qubits,
            quantum_depth,
            quantum_device,
            embedding=embedding,
            shots=shots,
            noise_probability=noise_probability,
        )
    else:
        raise ValueError(f"Unknown comparison model: {model_name}")
    return SharedEncoderClassifier(
        num_classes,
        encoder_dim=encoder_dim,
        n_qubits=n_qubits,
        dropout=dropout,
        feature_layer=features,
    )


def load_pretrained_encoder(model: SharedEncoderClassifier, checkpoint: str | Path) -> None:
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    encoder_state = {
        key.removeprefix("encoder."): value
        for key, value in state.items()
        if key.startswith("encoder.")
    }
    if not encoder_state:
        raise ValueError("Checkpoint does not contain encoder.* weights")
    model.encoder.load_state_dict(encoder_state)


def freeze_encoder(model: SharedEncoderClassifier, frozen: bool = True) -> None:
    for parameter in model.encoder.parameters():
        parameter.requires_grad = not frozen


def parameter_report(model: SharedEncoderClassifier) -> dict[str, int]:
    encoder = sum(parameter.numel() for parameter in model.encoder.parameters())
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    quantum_or_control = sum(parameter.numel() for parameter in model.feature_layer.parameters())
    return {
        "total": total,
        "trainable": trainable,
        "encoder": encoder,
        "feature_layer": quantum_or_control,
        "non_encoder": total - encoder,
    }
