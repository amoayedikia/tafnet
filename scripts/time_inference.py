#!/usr/bin/env python3
"""Inference cost of TAFNet-Full and CNN-LSTM on the same hardware (paper Section 6.3).

Times one pair (batch 1, two 128^3 volumes) per forward pass. Weights are randomly
initialised: timing does not depend on the weight values. Reports the whole model,
the shared encoder alone (two passes), and the fusion stage + classifier alone.

    PYTHONPATH=src python scripts/time_inference.py [output.json]      (from the repository root)
"""
import json, sys, time, numpy as np, torch
from tafnet.config import load_config
from tafnet.training.benchmarks import _build_model

cfg = load_config("configs/default.yaml")
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev, torch.cuda.get_device_name(0) if dev == "cuda" else "", "| torch", torch.__version__)
torch.manual_seed(0)
x1 = torch.randn(1, 1, 128, 128, 128, device=dev); x2 = torch.randn(1, 1, 128, 128, 128, device=dev)
WARM, N = 30, 300

def sync():
    if dev == "cuda": torch.cuda.synchronize()

def bench(fn, amp):
    ts = []
    with torch.no_grad(), torch.autocast(device_type=dev, enabled=amp):
        for i in range(WARM + N):
            sync(); t = time.perf_counter(); fn(); sync()
            if i >= WARM: ts.append((time.perf_counter() - t) * 1e3)
    ts = np.array(ts)
    return dict(median_ms=float(np.median(ts)), q25=float(np.percentile(ts, 25)), q75=float(np.percentile(ts, 75)))

out = {}
for name in ["TAFNet-Full", "CNN-LSTM"]:
    m = _build_model(name, cfg, None, dev).to(dev).eval()
    total = sum(p.numel() for p in m.parameters())
    enc = sum(p.numel() for p in m.encoder.parameters())
    r = dict(params_total=total, params_encoder=enc, params_after_encoder=total - enc)
    with torch.no_grad():
        f1 = m.encoder(x1); f2 = m.encoder(x2)
    if name == "TAFNet-Full":
        head = lambda: m.classifier(m.fusion(f1, f2).mean(dim=[2, 3, 4]))
    else:
        def head():
            seq = torch.stack([f1.mean(dim=[2, 3, 4]), f2.mean(dim=[2, 3, 4])], dim=1)
            return m.classifier(m.lstm(seq)[1][0][-1])
    for amp in (True, False):
        k = "amp" if amp else "fp32"
        r[f"full_{k}"] = bench(lambda: m(x1, x2), amp)
        r[f"encoder_x2_{k}"] = bench(lambda: (m.encoder(x1), m.encoder(x2)), amp)
        r[f"after_encoder_{k}"] = bench(head, amp)
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad(), torch.autocast(device_type=dev, enabled=True): m(x1, x2)
        r["peak_mem_mb_amp"] = torch.cuda.max_memory_allocated() / 2**20
    out[name] = r
    print(f"\n== {name}: parameters total {total:,} | encoder {enc:,} | after encoder {total - enc:,}")
    for k, v in r.items():
        if isinstance(v, dict): print(f"  {k:22s} median {v['median_ms']:8.3f} ms  [IQR {v['q25']:.3f}, {v['q75']:.3f}]")
    if "peak_mem_mb_amp" in r: print(f"  peak GPU memory (amp)  {r['peak_mem_mb_amp']:.0f} MB")
    del m
dst = sys.argv[1] if len(sys.argv) > 1 else "time_inference.json"
json.dump(out, open(dst, "w"), indent=1)
print("\nwrote", dst)
