#!/usr/bin/env python3
"""
score_oasis3_tafnet.py
======================
Score the 125 OASIS-3 scan pairs with the retrained (v4) TAFNet-Full models.

Uses all five fold checkpoints (TAFNet-Full_fold1..5.pth) and reports each
fold's probability plus the fold-ensemble mean -- the same ensemble rule used
for the ADNI held-out set in main-v4.

Crash-safe: every scored pair is appended to scores_log.csv straight away, and
a re-run skips pairs already in the log. The final wide CSV is rebuilt from the
log at the end.

Runs on the Mac or on a GCP VM -- just point the paths at the right places.

Usage (from the repo root):
    python3 scripts/07_score_oasis3_v4.py --pairs PAIRS.csv --vol-dir VOLS --ckpt-dir CKPTS --check
    python3 scripts/07_score_oasis3_v4.py --pairs PAIRS.csv --vol-dir VOLS --ckpt-dir CKPTS
Options:
    --pairs     CSV with OASISID, baseline_session, followup_session
    --vol-dir   folder of *_T1w_pp.nii.gz volumes (TAFNet-preprocessed)
    --ckpt-dir  folder holding TAFNet-Full_fold{1..5}.pth
    --repo      TAFNet code tree (the one containing src/ and configs/)
    --device    cpu (default) or mps
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import inspect
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent   # repo root (contains src/ and configs/)

N_FOLDS = 5


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", required=True, help="CSV with OASISID, baseline_session, followup_session")
    p.add_argument("--vol-dir", required=True, help="folder of *_T1w_pp.nii.gz volumes")
    p.add_argument("--ckpt-dir", required=True, help="folder with TAFNet-Full_fold{1..5}.pth")
    p.add_argument("--repo", default=None, help="TAFNet code tree (default: this repo)")
    p.add_argument("--out-dir", default="results_oasis3_v4")
    p.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    p.add_argument("--check", action="store_true",
                   help="verify everything and score only the first pair")
    return p.parse_args()


def pick_repo(arg):
    return Path(arg).expanduser() if arg else REPO_ROOT


def find_volume(vol_dir, session):
    """'OAS30007_d2722' -> sub-OAS30007_ses[s]-d2722[_run-01]_T1w_pp.nii.gz"""
    sid, day = session.split("_")
    hits = sorted(glob.glob(os.path.join(vol_dir, f"sub-{sid}_ses*-{day}_*T1w_pp.nii.gz")))
    if len(hits) != 1:
        raise FileNotFoundError(f"{session}: expected 1 volume, found {len(hits)}: {hits}")
    return hits[0]


def load_volume(path, torch, np, nib):
    """Same maths as tafnet.data.datasets._load_nifti_volume, but fails loudly
    instead of silently returning a zero volume."""
    data = nib.load(path).get_fdata(dtype=np.float32)
    if data.shape != (128, 128, 128):
        raise ValueError(f"{path}: shape {data.shape}, expected (128,128,128)")
    if data.max() > 1.0:
        data = (data - data.min()) / (data.max() - data.min() + 1e-8)
    return torch.from_numpy(data).unsqueeze(0).unsqueeze(0)      # (1,1,D,H,W)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def main():
    args = parse_args()

    # ---- imports (tell the user exactly what is missing) -------------------
    missing = []
    for mod in ("torch", "numpy", "pandas", "nibabel", "yaml", "sklearn"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        names = {"yaml": "pyyaml", "sklearn": "scikit-learn"}
        print("[X] Missing Python packages: " + ", ".join(missing))
        print("    Install them first, then re-run:")
        print("    python3 -m pip install " + " ".join(names.get(m, m) for m in missing))
        return 1
    import numpy as np
    import pandas as pd
    import nibabel as nib
    import torch
    import yaml

    repo = pick_repo(args.repo)
    sys.path.insert(0, str(repo / "src"))
    from tafnet.models.tafnet import TAFNet            # noqa: E402

    print(f"Code repo : {repo}")
    print(f"Volumes   : {args.vol_dir}")
    print(f"Checkpts  : {args.ckpt_dir}")
    print(f"torch     : {torch.__version__}   device: {args.device}")

    # ---- pairs + volume resolution -----------------------------------------
    pairs = pd.read_csv(args.pairs)
    pairs["pair_id"] = pairs["baseline_session"] + "__" + pairs["followup_session"]
    if pairs["pair_id"].duplicated().any():
        raise SystemExit("[X] duplicate pairs in the pairs CSV")
    pairs["baseline_file"] = pairs["baseline_session"].map(lambda s: find_volume(args.vol_dir, s))
    pairs["followup_file"] = pairs["followup_session"].map(lambda s: find_volume(args.vol_dir, s))
    print(f"[OK] {len(pairs)} pairs, all {len(set(pairs.baseline_file) | set(pairs.followup_file))} "
          f"volumes found on disk")

    # ---- models ------------------------------------------------------------
    cfg = yaml.safe_load(open(repo / "configs" / "default.yaml"))
    arch = cfg["architecture"]
    kw = dict(encoder_channels=tuple(arch["encoder_channels"]), use_dcca=bool(arch["use_dcca"]),
              feature_dim=int(arch["feature_dim"]), num_heads=int(arch["num_heads"]),
              dropout=float(arch["dropout"]), use_longitudinal=True, freeze_encoder=False,
              gate_mode=arch.get("gate_mode", "patient"),
              baseline_residual=bool(arch.get("baseline_residual", True)))
    kw = {k: v for k, v in kw.items() if k in inspect.signature(TAFNet.__init__).parameters}
    print(f"Architecture: {kw}")

    models, hashes = [], []
    for f in range(1, N_FOLDS + 1):
        ck = os.path.join(args.ckpt_dir, f"TAFNet-Full_fold{f}.pth")
        if not os.path.exists(ck):
            raise SystemExit(f"[X] checkpoint not found: {ck}")
        state = torch.load(ck, map_location="cpu")
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
        m = TAFNet(**kw)
        m.load_state_dict(state, strict=True)        # any architecture mismatch stops here
        m.eval().to(args.device)
        models.append(m)
        hashes.append(sha256(ck))
        print(f"[OK] fold {f} loaded strictly  ({sum(p.numel() for p in m.parameters()):,} params, "
              f"sha256 {hashes[-1]})")

    # Encoder frozen during fold training -> usually identical across folds.
    # If so, run it once per scan and reuse (5x faster); verified below.
    enc0 = models[0].encoder.state_dict()
    shared_encoder = all(
        all(torch.equal(enc0[k], mm.encoder.state_dict()[k]) for k in enc0) for mm in models[1:])
    print(f"Shared encoder across folds: {shared_encoder}")

    @torch.no_grad()
    def score_pair(t1, t2):
        probs, gates = [], []
        if shared_encoder:
            b1 = models[0].encoder(t1)
            b2 = models[0].encoder(t2)
        for mm in models:
            if not shared_encoder:
                b1, b2 = mm.encoder(t1), mm.encoder(t2)
            fused, aux = mm.fusion(b1, b2, return_aux=True)
            logit = mm.classifier(fused.mean(dim=[2, 3, 4]))
            probs.append(float(torch.sigmoid(logit).item()))
            g = aux.get("gate")
            gates.append([float(x) for x in g.flatten()[:3].cpu()] if g is not None
                         else [float("nan")] * 3)
        return probs, gates

    # ---- log / resume --------------------------------------------------------
    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, "scores_log.csv" if not args.check else "scores_check.csv")
    fields = (["pair_id"] + [f"p_fold{f}" for f in range(1, N_FOLDS + 1)]
              + [f"{g}_fold{f}" for f in range(1, N_FOLDS + 1) for g in ("alpha", "beta", "gamma")])
    done = set()
    if os.path.exists(log_path) and not args.check:
        done = set(pd.read_csv(log_path)["pair_id"])
        print(f"Resuming: {len(done)} pairs already scored")
    new_file = not os.path.exists(log_path) or args.check
    fh = open(log_path, "w" if args.check else "a", newline="")
    w = csv.DictWriter(fh, fieldnames=fields)
    if new_file:
        w.writeheader()

    todo = pairs if not args.check else pairs.head(1)
    t0 = time.time()
    n_new = 0
    for i, r in todo.iterrows():
        if r.pair_id in done:
            continue
        t1 = load_volume(r.baseline_file, torch, np, nib).to(args.device)
        t2 = load_volume(r.followup_file, torch, np, nib).to(args.device)
        probs, gates = score_pair(t1, t2)

        if args.check:
            with torch.no_grad():
                full = float(torch.sigmoid(models[0](t1, t2)).item())
            diff = abs(full - probs[0])
            print(f"[{'OK' if diff < 1e-5 else 'X'}] fast path vs model.forward, fold 1: "
                  f"{probs[0]:.6f} vs {full:.6f} (|diff| {diff:.2e})")
            if diff >= 1e-5:
                return 1

        row = {"pair_id": r.pair_id}
        for f in range(N_FOLDS):
            row[f"p_fold{f+1}"] = f"{probs[f]:.6f}"
            for j, g in enumerate(("alpha", "beta", "gamma")):
                row[f"{g}_fold{f+1}"] = f"{gates[f][j]:.6f}"
        w.writerow(row)
        fh.flush()
        n_new += 1
        el = time.time() - t0
        print(f"  [{len(done) + n_new:>3}/{len(pairs)}] {r.pair_id:<34} "
              f"ens={np.mean(probs):.4f}  ({el / n_new:.1f}s/pair)")
    fh.close()

    if args.check:
        print(f"\n[OK] Check passed. Wrote {log_path}. Now run without --check.")
        return 0

    # ---- final wide CSV ------------------------------------------------------
    log = pd.read_csv(log_path).drop_duplicates("pair_id", keep="last")
    out = pairs.merge(log, on="pair_id", how="left")
    if out["p_fold1"].isna().any():
        raise SystemExit(f"[X] {int(out['p_fold1'].isna().sum())} pairs unscored -- re-run to resume")
    pcols = [f"p_fold{f}" for f in range(1, N_FOLDS + 1)]
    out["tafnet_v4_ensemble"] = out[pcols].mean(axis=1)
    out["tafnet_v4_fold_sd"] = out[pcols].std(axis=1)
    for g in ("alpha", "beta", "gamma"):
        out[f"{g}_ensemble"] = out[[f"{g}_fold{f}" for f in range(1, N_FOLDS + 1)]].mean(axis=1)
    out["checkpoint_sha256"] = "|".join(hashes)
    final = os.path.join(args.out_dir, "OASIS3_125_tafnet_v4_scores.csv")
    out.drop(columns=["baseline_file", "followup_file"]).to_csv(final, index=False)

    s = out["tafnet_v4_ensemble"]
    print("\n" + "=" * 60)
    print(f"Scored {len(out)} pairs from {out.OASISID.nunique()} subjects")
    print(f"Ensemble score: median {s.median():.4f}, IQR [{s.quantile(.25):.4f}, "
          f"{s.quantile(.75):.4f}], {100 * (s < 0.01).mean():.1f}% below 0.01")
    if "pair_label_converter" in out and out.pair_label_converter.nunique() == 2:
        from sklearn.metrics import roc_auc_score
        print(f"Sanity AUC vs pair_label_converter (pooled pairs, not the analysis): "
              f"{roc_auc_score(out.pair_label_converter, s):.3f}")
    print(f"[OK] Wrote {final}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
