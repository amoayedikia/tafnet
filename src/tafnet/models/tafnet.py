"""
Temporal Adaptive Fusion Network (TAFNet) — proposed method.

Three-Branch Temporal Fusion (TBT) module:
    Branch A: cross-temporal attention (Q from T2, K/V from T1)
    Branch B: temporal difference (T2 - T1)
    Branch C: concatenation + projection
    Adaptive gate: softmax over branches at each spatial position.

TAFNet wraps a shared JDACEncoder3D, the TBT fusion module, and a GAP +
2-layer classifier head. Supports an `initial_only` ablation mode that
bypasses the fusion module (useful as a same-architecture single-timepoint
baseline).
"""
from __future__ import annotations

import os
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import JDACEncoder3D


class ThreeBranchTemporalFusion(nn.Module):
    """Three-branch fusion with adaptive per-position gating."""

    def __init__(self, feature_dim: int = 128, num_heads: int = 8,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(feature_dim)
        self.norm_k = nn.LayerNorm(feature_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.concat_proj = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(inplace=True),
        )
        self.gate_logic = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, 3),
        )

    def forward(self, feat_t1: torch.Tensor, feat_t2: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = feat_t1.shape
        n = d * h * w

        t1_seq = feat_t1.reshape(b, c, n).permute(0, 2, 1)  # (B, N, C)
        t2_seq = feat_t2.reshape(b, c, n).permute(0, 2, 1)

        # Branch A: cross-temporal attention (T2 queries T1)
        q = self.norm_q(t2_seq)
        k = self.norm_k(t1_seq)
        attn_out, _ = self.cross_attn(query=q, key=k, value=t1_seq)

        # Branch B: temporal difference
        diff_out = t2_seq - t1_seq

        # Branch C: concatenation + projection
        concat_out = self.concat_proj(torch.cat([t1_seq, t2_seq], dim=-1))

        # Adaptive gate over (A, B, C) at every spatial position
        combined = torch.cat([attn_out, diff_out, concat_out], dim=-1)
        gate = F.softmax(self.gate_logic(combined), dim=-1)
        g1, g2, g3 = gate[..., 0:1], gate[..., 1:2], gate[..., 2:3]
        fused_seq = g1 * attn_out + g2 * diff_out + g3 * concat_out

        return fused_seq.permute(0, 2, 1).reshape(b, c, d, h, w)


class TAFNet(nn.Module):
    """
    TAFNet: JDACEncoder3D + Three-Branch Temporal Fusion + GAP + classifier.

    Modes:
        use_longitudinal=True  : full model (T1 + T2 -> fusion -> classify)
        use_longitudinal=False : initial-only ablation (T1 -> GAP -> classify)
    """

    def __init__(
        self,
        encoder_channels: Sequence[int] = (16, 32, 64, 128, 128),
        use_dcca: bool = True,
        feature_dim: int = 128,
        num_heads: int = 8,
        dropout: float = 0.3,
        use_longitudinal: bool = True,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()
        self.use_longitudinal = use_longitudinal
        self.freeze_encoder = freeze_encoder

        self.encoder = JDACEncoder3D(
            in_ch=1, channels=encoder_channels, use_dcca=use_dcca,
        )
        if self.use_longitudinal:
            self.fusion = ThreeBranchTemporalFusion(
                feature_dim=feature_dim, num_heads=num_heads, dropout=0.1,
            )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

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
        b1 = self.encoder(x_t1)
        if self.use_longitudinal and x_t2 is not None:
            b2 = self.encoder(x_t2)
            fused = self.fusion(b1, b2)
        else:
            fused = b1
        pooled = fused.mean(dim=[2, 3, 4])
        return self.classifier(pooled)

    def count_parameters(self, only_trainable: bool = True) -> int:
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
