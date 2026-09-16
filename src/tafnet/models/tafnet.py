"""
Temporal Adaptive Fusion Network (TAFNet) — proposed method.

Implements Section 4 of the manuscript as written:

    Branch A: temporal difference      Δf = f_M12 − f_BL          (Eq 13)
    Branch B: cross-temporal attention Q←f_BL, K,V←f_M12, H=4     (Eq 14–15)
    Branch C: concatenation + 1×1×1 projection                    (Eq 16)
    Adaptive Temporal Gate: softmax MLP over [GAP(f_BL); GAP(f_M12)],
                            256→64→3, ONE triple per patient      (Eq 17–18)
    f_fused = α·Δf + β·Att + γ·f_cat                              (Eq 19)
    f_out   = f_fused + f_BL          baseline residual           (Eq 20)

Branch order is (α, β, γ) = (difference, attention, concat), matching
Algorithm 4 line 6. The previous implementation had index 0 = attention, which
made every α/β/γ statement in the paper refer to the wrong branch.

`gate_mode` selects between the specified gate and the previous one:

    "patient"  — Algorithm 4 / Eq 17–18. Conditioned on the two global average
                 descriptors; one (α, β, γ) per patient; ~16.6K parameters.
    "position" — the earlier implementation. Conditioned on the three branch
                 OUTPUTS at each of the 512 spatial positions; one triple per
                 position. Retained so the two can be compared as an ablation
                 rather than silently replaced.

Both modes are otherwise identical, so a "patient" vs "position" comparison
isolates the gate.

TAFNet wraps a shared JDACEncoder3D, this fusion module, and a GAP + 2-layer
classifier head. `initial_only` bypasses fusion for the single-timepoint
ablation. `forward(..., return_aux=True)` returns the gate coefficients and the
per-head attention matrices, which is what Algorithm 5 needs.
"""
from __future__ import annotations

import os
import re
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import JDACEncoder3D

GATE_MODES = ("patient", "position")
#: Index → branch, per Algorithm 4 line 6. Do not reorder without updating §5.4.
BRANCH_ORDER = ("difference", "attention", "concat")


def normalise_branches(value) -> Tuple[str, ...]:
    """
    Coerce a `branches` specification into a validated tuple in BRANCH_ORDER.

    Accepts None (= all three), a sequence of names, or a comma-separated
    string (which is what `--override architecture.branches=diff,concat`
    delivers, since the override parser only produces scalars). Short forms
    "diff" and "attn" are accepted.
    """
    if value is None:
        return BRANCH_ORDER
    if isinstance(value, str):
        value = [v for v in re.split(r"[,\s]+", value) if v]
    alias = {"diff": "difference", "attn": "attention",
             "concatenation": "concat", "cat": "concat"}
    names = [alias.get(str(v).strip().lower(), str(v).strip().lower())
             for v in value]
    unknown = [n for n in names if n not in BRANCH_ORDER]
    if unknown:
        raise ValueError(
            f"unknown branch(es) {unknown}; valid names are {BRANCH_ORDER}")
    out = tuple(b for b in BRANCH_ORDER if b in names)
    if not out:
        raise ValueError("at least one fusion branch must be enabled")
    return out


class ThreeBranchTemporalFusion(nn.Module):
    """Three-branch temporal fusion with an Adaptive Temporal Gate."""

    def __init__(
        self,
        feature_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
        gate_mode: str = "patient",
        gate_hidden: int = 64,
        baseline_residual: bool = True,
        branches: Optional[Sequence[str] | str] = None,
    ) -> None:
        super().__init__()
        if gate_mode not in GATE_MODES:
            raise ValueError(f"gate_mode must be one of {GATE_MODES}, got {gate_mode!r}")
        if feature_dim % num_heads != 0:
            raise ValueError(
                f"feature_dim {feature_dim} not divisible by num_heads {num_heads}"
            )
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads      # d_k = 32 at H = 4
        self.gate_mode = gate_mode
        self.baseline_residual = baseline_residual

        # Branch ablation. `branches` names the branches that stay in the
        # mixture; the rest have their gate logits masked to -inf BEFORE the
        # softmax, so an ablated model is exactly the renormalised mixture over
        # the branches that remain (zeroing outputs after the softmax would
        # leave probability mass stranded on a dead branch — a different
        # experiment). The disabled branches are still computed; they simply
        # receive weight 0. Not registered in the state_dict, so checkpoints
        # remain interchangeable across variants.
        self.branches = normalise_branches(branches)
        self.register_buffer(
            "branch_mask",
            torch.tensor([b in self.branches for b in BRANCH_ORDER]),
            persistent=False,
        )

        # Branch B — cross-attention. Queries come from the BASELINE and
        # keys/values from the follow-up, per Eq 14. §4.2.2 argues for exactly
        # this direction: each baseline position asks every follow-up position
        # how it has changed.
        self.norm_q = nn.LayerNorm(feature_dim)
        self.norm_kv = nn.LayerNorm(feature_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )

        # Branch C — Eq 16. A Linear over the channel axis of (B, N, C) is the
        # same map as a 1×1×1 Conv3d over (B, C, D, H, W). No activation: Eq 16
        # specifies the convolution alone.
        self.concat_proj = nn.Linear(feature_dim * 2, feature_dim)

        # Adaptive Temporal Gate
        if gate_mode == "patient":
            gate_in = feature_dim * 2          # [GAP(f_BL); GAP(f_M12)] ∈ R^256
        else:
            gate_in = feature_dim * 3          # [Δf; Att; f_cat] at each position
        self.gate_logic = nn.Sequential(
            nn.Linear(gate_in, gate_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(gate_hidden, 3),
        )

    # -- gate ---------------------------------------------------------------

    def _gate(self, feat_t1, feat_t2, branch_seq) -> torch.Tensor:
        """
        Return softmax gate coefficients.

            "patient"  -> (B, 1, 3)   broadcast across all spatial positions
            "position" -> (B, N, 3)   one triple per position
        """
        if self.gate_mode == "patient":
            z_bl = feat_t1.mean(dim=[2, 3, 4])          # GAP  -> (B, 128)
            z_m12 = feat_t2.mean(dim=[2, 3, 4])
            g = torch.cat([z_bl, z_m12], dim=-1)        # (B, 256)
            logits = self.gate_logic(g).unsqueeze(1)    # (B, 1, 3)
        else:
            logits = self.gate_logic(branch_seq)        # (B, N, 3)
        if not bool(self.branch_mask.all()):
            logits = logits.masked_fill(
                ~self.branch_mask.view(1, 1, 3), float("-inf"))
        return F.softmax(logits, dim=-1)

    # -- forward ------------------------------------------------------------

    def forward(
        self,
        feat_t1: torch.Tensor,
        feat_t2: torch.Tensor,
        return_aux: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        b, c, d, h, w = feat_t1.shape
        n = d * h * w                                   # 512 at the 8³ bottleneck

        t1_seq = feat_t1.reshape(b, c, n).permute(0, 2, 1)   # (B, N, C)
        t2_seq = feat_t2.reshape(b, c, n).permute(0, 2, 1)

        # Branch A — temporal difference (Eq 13)
        diff_out = t2_seq - t1_seq

        # Branch B — cross-temporal attention (Eq 14–15)
        q = self.norm_q(t1_seq)                          # queries  <- baseline
        kv = self.norm_kv(t2_seq)                        # keys/values <- follow-up
        attn_out, attn_w = self.cross_attn(
            query=q, key=kv, value=t2_seq,
            need_weights=return_aux, average_attn_weights=False,
        )

        # Branch C — concatenation + 1×1×1 projection (Eq 16)
        concat_out = self.concat_proj(torch.cat([t1_seq, t2_seq], dim=-1))

        # Adaptive Temporal Gate (Eq 17–18); order is (difference, attention, concat)
        branch_seq = torch.cat([diff_out, attn_out, concat_out], dim=-1)
        gate = self._gate(feat_t1, feat_t2, branch_seq)
        alpha, beta, gamma = gate[..., 0:1], gate[..., 1:2], gate[..., 2:3]

        # Weighted mixture (Eq 19)
        fused_seq = alpha * diff_out + beta * attn_out + gamma * concat_out

        # Baseline residual (Eq 20 / Alg 3 line 14) — preserves static anatomy,
        # which subtraction discards (§4.2.1)
        if self.baseline_residual:
            fused_seq = fused_seq + t1_seq

        fused = fused_seq.permute(0, 2, 1).reshape(b, c, d, h, w)
        if not return_aux:
            return fused

        aux = {
            # (B, 3) per patient. In "position" mode this is the mean over the
            # 512 positions; `gate_full` keeps the per-position tensor.
            "gate": gate.mean(dim=1) if gate.shape[1] > 1 else gate.squeeze(1),
            "gate_full": gate,
            "gate_mode": self.gate_mode,
            "branch_order": BRANCH_ORDER,
            "branches_enabled": self.branches,
            # (B, H, N, N) per-head attention — Algorithm 5 line 2
            "attn_weights": attn_w,
        }
        return fused, aux


class TAFNet(nn.Module):
    """JDACEncoder3D + three-branch temporal fusion + GAP + classifier."""

    def __init__(
        self,
        encoder_channels: Sequence[int] = (16, 32, 64, 128, 128),
        use_dcca: bool = True,
        feature_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.3,
        use_longitudinal: bool = True,
        freeze_encoder: bool = False,
        gate_mode: str = "patient",
        baseline_residual: bool = True,
        branches: Optional[Sequence[str] | str] = None,
    ) -> None:
        super().__init__()
        self.use_longitudinal = use_longitudinal
        self.freeze_encoder = freeze_encoder
        self.gate_mode = gate_mode

        self.encoder = JDACEncoder3D(
            in_ch=1, channels=encoder_channels, use_dcca=use_dcca,
        )
        if self.use_longitudinal:
            self.fusion = ThreeBranchTemporalFusion(
                feature_dim=feature_dim, num_heads=num_heads, dropout=0.1,
                gate_mode=gate_mode, baseline_residual=baseline_residual,
                branches=branches,
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
        # strict=True: a silent partial load is what produced the §5.4 defect.
        self.encoder.load_state_dict(encoder_state, strict=True)
        print(f"  Loaded encoder from: {checkpoint_path}")
        if self.freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            print("  Encoder FROZEN")
        return True

    def forward(
        self,
        x_t1: torch.Tensor,
        x_t2: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ):
        b1 = self.encoder(x_t1)
        aux: Dict = {}
        if self.use_longitudinal and x_t2 is not None:
            b2 = self.encoder(x_t2)
            if return_aux:
                fused, aux = self.fusion(b1, b2, return_aux=True)
            else:
                fused = self.fusion(b1, b2)
        else:
            fused = b1
        pooled = fused.mean(dim=[2, 3, 4])
        logits = self.classifier(pooled)
        if return_aux:
            aux["bottleneck_bl"] = b1
            return logits, aux
        return logits

    # -- Algorithm 5 --------------------------------------------------------

    @torch.no_grad()
    def interpret(self, x_t1: torch.Tensor, x_t2: torch.Tensor,
                  out_size: int = 128) -> Dict[str, torch.Tensor]:
        """
        Algorithm 5: spatial attention map and gate profile.

        Head-average the attention matrices (line 3), sum received attention per
        query position (line 4), reshape to the 8³ grid (line 5), upsample
        trilinearly to `out_size`³ (line 6) and normalise to [0, 1] (line 7).

        Returns {"attention_map": (B, S, S, S), "gate": (B, 3),
                 "prediction": (B,)}. `gate` is ordered (α, β, γ) =
        (difference, attention, concat).
        """
        self.eval()
        logits, aux = self.forward(x_t1, x_t2, return_aux=True)
        attn = aux["attn_weights"]                 # (B, H, N, N)
        if attn is None:
            raise RuntimeError("attention weights unavailable — needs use_longitudinal")

        a_bar = attn.mean(dim=1)                   # (B, N, N)   line 3
        received = a_bar.sum(dim=1)                # (B, N)      line 4
        b, n = received.shape
        g = round(n ** (1 / 3))
        if g ** 3 != n:
            raise RuntimeError(f"{n} tokens is not a cube; cannot reshape to a grid")
        grid = received.reshape(b, 1, g, g, g)     # line 5
        up = F.interpolate(grid, size=(out_size,) * 3,
                           mode="trilinear", align_corners=False)
        flat = up.reshape(b, -1)
        lo = flat.min(dim=1).values.view(b, 1, 1, 1, 1)
        hi = flat.max(dim=1).values.view(b, 1, 1, 1, 1)
        up = (up - lo) / (hi - lo + 1e-8)          # line 7

        return {
            "attention_map": up.squeeze(1),
            "gate": aux["gate"],
            "prediction": torch.sigmoid(logits).flatten(),
        }

    def count_parameters(self, only_trainable: bool = True) -> int:
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
