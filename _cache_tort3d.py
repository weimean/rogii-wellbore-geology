"""Cache proper 3D Q-tortuosity features per eval row, aligned to /tmp/rogii_feats.npz.
mycarta's claimed best single feature (-0.107): cumulative high-frequency oscillation of the
3D wellbore trajectory (active steering signal). Our existing `tort` is 1D (rolling std of dZ);
this uses the full (X,Y,Z) tangent-vector dogleg.

Per eval row: dls (3D dogleg angle/step), rolling sum at 31/151 (local/broad steering),
rolling std at 51, and cumulative dogleg from the PS point (total eval-zone steering).
Saves /tmp/rogii_tort3d.npz. Run: uv run python -u _cache_tort3d.py
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import pandas as pd
from pathlib import Path
from src.data import load_split, TARGET_INPUT

t0 = time.time()
def log(m): print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)

D = np.load("/tmp/rogii_feats.npz"); GID = D["gidx"]
wells = load_split(Path("data/raw/train"))

cols = {k: [] for k in ["dls", "tort31", "tort151", "dlsstd51", "cumtort"]}
for w in wells:
    h = w.horizontal; em = w.eval_mask.to_numpy(); kn = h[h[TARGET_INPUT].notna()]
    if em.sum() == 0 or len(kn) < 200:
        continue
    X = h["X"].to_numpy(float); Y = h["Y"].to_numpy(float); Z = h["Z"].to_numpy(float)
    P = np.column_stack([X, Y, Z])
    t = np.diff(P, axis=0)                       # tangent segments
    nrm = np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
    tu = t / nrm
    dot = np.clip(np.sum(tu[1:] * tu[:-1], axis=1), -1, 1)
    dls = np.concatenate([[0.0], np.arccos(dot), [0.0]])   # angle/step, len == n rows
    s = pd.Series(dls)
    tort31 = s.rolling(31, center=True, min_periods=1).sum().to_numpy()
    tort151 = s.rolling(151, center=True, min_periods=1).sum().to_numpy()
    dlsstd51 = s.rolling(51, center=True, min_periods=1).std().fillna(0).to_numpy()
    ps_i = int(np.where(em)[0][0])               # first eval row = PS
    cum = np.cumsum(dls)
    cumtort = cum - cum[ps_i]                     # cumulative steering since PS
    ei = np.where(em)[0]
    for k, arr in [("dls", dls), ("tort31", tort31), ("tort151", tort151),
                   ("dlsstd51", dlsstd51), ("cumtort", cumtort)]:
        cols[k].append(arr[ei])

out = {k: np.concatenate(v) for k, v in cols.items()}
assert len(out["dls"]) == len(GID), f"misalign {len(out['dls'])} vs {len(GID)}"
np.savez_compressed("/tmp/rogii_tort3d.npz", **out)
log(f"saved /tmp/rogii_tort3d.npz rows={len(out['dls'])}")
