"""
Benchmark methods for MCI-to-AD conversion prediction.

ResNet3D18           — single-timepoint 3D ResNet-18
DenseNet3D121        — single-timepoint 3D DenseNet-121 (reduced growth_rate)
SiameseCNNSubtract   — JDACEncoder3D twin + (feat_T2 - feat_T1) + GAP + FC
CNNLSTM3D            — JDACEncoder3D twin + 1-layer LSTM on (T1, T2) sequence

All longitudinal models implement `load_pretrained_encoder(path, device)` to
optionally load the Phase 4 checkpoint.
"""
from __future__ import annotations

import os
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import JDACEncoder3D


# ===========================================================================
# 3D ResNet-18 (single timepoint)
# ===========================================================================

class _BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.shortcut: nn.Module = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm3d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ResNet3D18(nn.Module):
    """3D ResNet-18 operating on the initial scan only."""

    def __init__(self, in_ch: int = 1, num_classes: int = 1,
                 dropout: float = 0.3) -> None:
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv3d(in_ch, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.maxpool = nn.MaxPool3d(3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )
        self.use_longitudinal = False

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(_BasicBlock3D(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x_t1: torch.Tensor,
                x_t2: torch.Tensor | None = None) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x_t1)))
        x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x).view(x.size(0), -1)
        return self.classifier(x)

    def count_parameters(self, only_trainable: bool = True) -> int:
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


# ===========================================================================
# 3D DenseNet-121 (single timepoint, reduced)
# ===========================================================================

class _DenseLayer3D(nn.Module):
    def __init__(self, in_ch: int, growth_rate: int, bn_size: int = 4) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm3d(in_ch)
        self.conv1 = nn.Conv3d(in_ch, bn_size * growth_rate, 1, bias=False)
        self.bn2 = nn.BatchNorm3d(bn_size * growth_rate)
        self.conv2 = nn.Conv3d(bn_size * growth_rate, growth_rate, 3,
                               padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        return torch.cat([x, out], dim=1)


class _DenseBlock3D(nn.Module):
    def __init__(self, in_ch: int, growth_rate: int, num_layers: int) -> None:
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(_DenseLayer3D(in_ch + i * growth_rate, growth_rate))
        self.layers = nn.Sequential(*layers)
        self.out_channels = in_ch + num_layers * growth_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class _Transition3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.bn = nn.BatchNorm3d(in_ch)
        self.conv = nn.Conv3d(in_ch, out_ch, 1, bias=False)
        self.pool = nn.AvgPool3d(2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.conv(F.relu(self.bn(x))))


class DenseNet3D121(nn.Module):
    """
    3D DenseNet-121 operating on the initial scan only.
    Reduced: growth_rate=16, compression=0.5 for memory.
    """

    def __init__(self, in_ch: int = 1, num_classes: int = 1,
                 growth_rate: int = 16,
                 block_config: Sequence[int] = (6, 12, 24, 16),
                 compression: float = 0.5, dropout: float = 0.3) -> None:
        super().__init__()
        num_init_features = 64
        self.features = nn.Sequential(
            nn.Conv3d(in_ch, num_init_features, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(3, stride=2, padding=1),
        )
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock3D(num_features, growth_rate, num_layers)
            self.features.add_module(f"denseblock{i+1}", block)
            num_features = block.out_channels
            if i != len(block_config) - 1:
                trans = _Transition3D(num_features, int(num_features * compression))
                self.features.add_module(f"transition{i+1}", trans)
                num_features = int(num_features * compression)
        self.features.add_module("norm_final", nn.BatchNorm3d(num_features))
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )
        self.use_longitudinal = False

    def forward(self, x_t1: torch.Tensor,
                x_t2: torch.Tensor | None = None) -> torch.Tensor:
        x = self.features(x_t1)
        x = F.relu(x)
        x = self.avgpool(x).view(x.size(0), -1)
        return self.classifier(x)

    def count_parameters(self, only_trainable: bool = True) -> int:
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


# ===========================================================================
# Siamese CNN with subtraction fusion (longitudinal)
# ===========================================================================

class SiameseCNNSubtract(nn.Module):
    """
    Shared encoder + temporal difference fusion (f(T2) - f(T1)).
    Common longitudinal baseline (e.g. Qiu et al. 2023).
    """

    def __init__(self, encoder_channels: Sequence[int] = (16, 32, 64, 128, 128),
                 use_dcca: bool = True, feature_dim: int = 128,
                 dropout: float = 0.3, freeze_encoder: bool = False) -> None:
        super().__init__()
        self.freeze_encoder = freeze_encoder
        self.encoder = JDACEncoder3D(in_ch=1, channels=encoder_channels,
                                     use_dcca=use_dcca)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        self.use_longitudinal = True

    def load_pretrained_encoder(self, checkpoint_path: str,
                                device: str = "cpu") -> bool:
        if not os.path.exists(checkpoint_path):
            print(f"  Checkpoint not found: {checkpoint_path}")
            return False
        ckpt = torch.load(checkpoint_path, map_location=device)
        encoder_state = {k.replace("encoder.", ""): v
                         for k, v in ckpt.items() if k.startswith("encoder.")}
        self.encoder.load_state_dict(encoder_state)
        print(f"  Loaded encoder from: {checkpoint_path}")
        if self.freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            print("  Encoder FROZEN")
        return True

    def forward(self, x_t1: torch.Tensor,
                x_t2: torch.Tensor | None = None) -> torch.Tensor:
        feat_t1 = self.encoder(x_t1)
        if x_t2 is not None:
            feat_t2 = self.encoder(x_t2)
            diff = feat_t2 - feat_t1
        else:
            diff = feat_t1
        pooled = diff.mean(dim=[2, 3, 4])
        return self.classifier(pooled)

    def count_parameters(self, only_trainable: bool = True) -> int:
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


# ===========================================================================
# CNN + LSTM (longitudinal)
# ===========================================================================

class CNNLSTM3D(nn.Module):
    """
    Shared 3D encoder + 1-layer LSTM over the (T1, T2) feature sequence.
    Used in Aghajanian et al. (2025) and similar.
    """

    def __init__(self, encoder_channels: Sequence[int] = (16, 32, 64, 128, 128),
                 use_dcca: bool = True, feature_dim: int = 128,
                 hidden_dim: int = 64, num_layers: int = 1,
                 dropout: float = 0.3, freeze_encoder: bool = False) -> None:
        super().__init__()
        self.freeze_encoder = freeze_encoder
        self.feature_dim = feature_dim
        self.encoder = JDACEncoder3D(in_ch=1, channels=encoder_channels,
                                     use_dcca=use_dcca)
        self.lstm = nn.LSTM(
            input_size=feature_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
        self.use_longitudinal = True

    def load_pretrained_encoder(self, checkpoint_path: str,
                                device: str = "cpu") -> bool:
        if not os.path.exists(checkpoint_path):
            print(f"  Checkpoint not found: {checkpoint_path}")
            return False
        ckpt = torch.load(checkpoint_path, map_location=device)
        encoder_state = {k.replace("encoder.", ""): v
                         for k, v in ckpt.items() if k.startswith("encoder.")}
        self.encoder.load_state_dict(encoder_state)
        print(f"  Loaded encoder from: {checkpoint_path}")
        if self.freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            print("  Encoder FROZEN")
        return True

    def forward(self, x_t1: torch.Tensor,
                x_t2: torch.Tensor | None = None) -> torch.Tensor:
        feat_t1 = self.encoder(x_t1).mean(dim=[2, 3, 4])     # (B, C)
        if x_t2 is not None:
            feat_t2 = self.encoder(x_t2).mean(dim=[2, 3, 4])
            seq = torch.stack([feat_t1, feat_t2], dim=1)      # (B, 2, C)
        else:
            seq = feat_t1.unsqueeze(1)                        # (B, 1, C)
        _out, (h_n, _c_n) = self.lstm(seq)
        final_hidden = h_n[-1]
        return self.classifier(final_hidden)

    def count_parameters(self, only_trainable: bool = True) -> int:
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
