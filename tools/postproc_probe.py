"""Postprocessing lever (notes/23): the sub-9 notebooks all shape the OUTPUT — per-well Savitzky-Golay
smoothing + a distance ramp (1-e^{-md_since/tau}) on the correction + delta-target. We track as well as
they do (notes/22), so output-shaping is the untested honest lever. Tested on the CACHED OOF (no retrain).

    uv run python -u tools/postproc_probe.py
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.signal import savgol_filter
from src.validation import score

d = np.load("/tmp/rogii_meta_oof.npz")
truth, anchor, fusion, g = d["truth"], d["anchor"], d["fusion"], d["g"]
gbm, mlp, mds = d["oof_gbm"], d["oof_mlp"], d["md_since"]
base = 0.75 * (anchor + 0.7 * gbm + 0.3 * mlp) + 0.25 * fusion
uniq, first = np.unique(g, return_index=True)
counts = np.diff(np.append(first, len(g)))


def per_well_savgol(x, win, poly):
    out = x.copy()
    for f, c in zip(first, counts):
        if c < win:
            continue
        w = win if win % 2 == 1 else win + 1
        w = min(w, c if c % 2 == 1 else c - 1)
        if w <= poly:
            continue
        out[f:f + c] = savgol_filter(x[f:f + c], w, poly)
    return out


def ramp_pred(pred, tau):
    r = 1.0 - np.exp(-mds / tau)
    return anchor + r * (pred - anchor)


print(f"base OOF pooled RMSE: {score(truth, base):.4f}\n", flush=True)

print("=== per-well Savitzky-Golay on the FINAL pred ===", flush=True)
for win in (7, 11, 17, 31, 51):
    for poly in (2, 3):
        p = per_well_savgol(base, win, poly)
        print(f"  savgol(win={win:2d}, poly={poly}): {score(truth, p):.4f}", flush=True)

print("\n=== per-well Savitzky-Golay on the CORRECTION (pred-anchor), added back ===", flush=True)
corr = base - anchor
for win in (11, 17, 31):
    for poly in (2, 3):
        p = anchor + per_well_savgol(corr, win, poly)
        print(f"  savgol_corr(win={win:2d}, poly={poly}): {score(truth, p):.4f}", flush=True)

print("\n=== distance ramp on the correction: anchor + (1-e^{-md_since/tau})*(pred-anchor) ===", flush=True)
for tau in (20, 50, 100, 200, 500):
    print(f"  ramp(tau={tau:3d}): {score(truth, ramp_pred(base, tau)):.4f}", flush=True)

print("\n=== best savgol + ramp combined ===", flush=True)
for win in (17, 31):
    for tau in (100, 200, 500):
        p = ramp_pred(per_well_savgol(base, win, 3), tau)
        print(f"  savgol(win={win},p3)+ramp(tau={tau}): {score(truth, p):.4f}", flush=True)
