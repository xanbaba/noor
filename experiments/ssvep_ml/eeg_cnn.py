"""Minimal temporal CNN for single-trial EEG windows (optional torch dependency)."""

from __future__ import annotations

import torch
from torch import nn


class TinyEegCNN(nn.Module):
    """Lightweight conv stack: ``(B, C, T)`` logits for K classes."""

    def __init__(self, n_channels: int, n_time: int, n_classes: int = 2) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.n_time = n_time
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 25), padding=(0, 12)),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.Conv2d(16, 32, kernel_size=(n_channels, 1)),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((1, 16)),
            nn.Flatten(),
            nn.Linear(32 * 16, 64),
            nn.ELU(),
            nn.Dropout(0.35),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        x = x.unsqueeze(1)
        return self.net(x)
