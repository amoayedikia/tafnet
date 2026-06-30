"""
Shared 3D encoder backbone (JDACEncoder3D) used by all longitudinal methods.

Provides:
    ConvBlock3D       — Conv3d + BN + LeakyReLU
    DoubleConvBlock3D — two ConvBlock3D in series
    DCCA3D            — Dense Context Channel Attention block
    JDACEncoder3D     — five-level encoder with optional DCCA, ending in a
                        (B, 128, 8, 8, 8) bottleneck for 128^3 input.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class ConvBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.bn = nn.BatchNorm3d(out_ch)
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DoubleConvBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv1 = ConvBlock3D(in_ch, out_ch)
        self.conv2 = ConvBlock3D(out_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1(x))


class DCCA3D(nn.Module):
    """Dense Context Channel Attention — 3D variant."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.context = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        reduced = max(channels // reduction, 8)
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ctx = self.context(x)
        b, c = ctx.shape[:2]
        y = self.gap(ctx).view(b, c)
        return x * self.fc(y).view(b, c, 1, 1, 1)


class JDACEncoder3D(nn.Module):
    """
    Joint Denoising and Artifact Correction Encoder.
    Five-level 3D U-Net-style encoder with optional channel attention at
    each level. Output bottleneck shape for 128^3 input: (B, 128, 8, 8, 8).
    """

    def __init__(
        self,
        in_ch: int = 1,
        channels: Sequence[int] = (16, 32, 64, 128, 128),
        use_dcca: bool = True,
    ) -> None:
        super().__init__()
        self.channels = list(channels)
        self.enc1 = DoubleConvBlock3D(in_ch, channels[0])
        self.enc2 = DoubleConvBlock3D(channels[0], channels[1])
        self.enc3 = DoubleConvBlock3D(channels[1], channels[2])
        self.enc4 = DoubleConvBlock3D(channels[2], channels[3])
        self.enc5 = DoubleConvBlock3D(channels[3], channels[4])
        self.pool = nn.MaxPool3d(2, 2)
        self.use_dcca = use_dcca
        if use_dcca:
            self.dcca1 = DCCA3D(channels[0])
            self.dcca2 = DCCA3D(channels[1])
            self.dcca3 = DCCA3D(channels[2])
            self.dcca4 = DCCA3D(channels[3])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        if self.use_dcca:
            e1 = self.dcca1(e1)
        e2 = self.enc2(self.pool(e1))
        if self.use_dcca:
            e2 = self.dcca2(e2)
        e3 = self.enc3(self.pool(e2))
        if self.use_dcca:
            e3 = self.dcca3(e3)
        e4 = self.enc4(self.pool(e3))
        if self.use_dcca:
            e4 = self.dcca4(e4)
        bottleneck = self.enc5(self.pool(e4))
        return bottleneck
