from __future__ import annotations

import torch
from torch import nn


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=7, stride=stride, padding=3, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels and stride == 1
            else nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        )
        self.activation = nn.GELU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.layers(inputs) + self.skip(inputs))


class ECGEncoder(nn.Module):
    """A 12-lead 1D CNN that returns a compact recording-level embedding."""

    def __init__(self, in_channels: int = 12, latent_dim: int = 32, dropout: float = 0.25) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(32),
            nn.GELU(),
            ResidualBlock1D(32, 32, stride=2),
            ResidualBlock1D(32, 64, stride=2),
            ResidualBlock1D(64, 128, stride=2),
            nn.AdaptiveAvgPool1d(1),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError("Expected ECG input with shape [batch, leads, time]")
        return self.projection(self.features(inputs))


class ClassicalECGClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        *,
        in_channels: int = 12,
        latent_dim: int = 32,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.encoder = ECGEncoder(in_channels, latent_dim, dropout)
        self.classifier = nn.Linear(latent_dim, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(inputs))

