"""Phase 4 model — encoder + GAP + classifier for cross-sectional pretraining."""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from .encoder import JDACEncoder3D


class Phase4Model(nn.Module):
    """JDACEncoder3D + GAP + 2-layer classifier, trained on CN vs AD."""

    def __init__(
        self,
        encoder_channels: Sequence[int] = (16, 32, 64, 128, 128),
        use_dcca: bool = True,
        feature_dim: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.encoder = JDACEncoder3D(in_ch=1, channels=encoder_channels,
                                     use_dcca=use_dcca)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bottleneck = self.encoder(x)
        pooled = bottleneck.mean(dim=[2, 3, 4])
        return self.classifier(pooled)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
