from __future__ import annotations

import torch
from torch import nn


class SmallForcCNN(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        output_dim: int = 1,
        base_channels: int = 32,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        c = base_channels
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, c, kernel_size=3, padding=1),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(c, c * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(c * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(c * 2, c * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(c * 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(c * 4, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def build_model(config: dict) -> nn.Module:
    model_cfg = config["model"]
    return SmallForcCNN(
        input_channels=int(model_cfg["input_channels"]),
        output_dim=int(model_cfg["output_dim"]),
        base_channels=int(model_cfg["base_channels"]),
        dropout=float(model_cfg["dropout"]),
    )
