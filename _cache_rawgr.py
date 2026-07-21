"""RAW-GR sequence channels (notes/27 'last breath'): the un-reduced correlation surface. Rich features
give hgr - tw_gr(HMM+off) at only 11 hand-picked offsets; here we give the FINE grid (41 offsets around
the HMM path) so a from-scratch sequence model can learn its OWN reduction, plus the raw typewell reference
curve at those offsets (so it can learn calibration) and raw multi-scale horizontal GR. Aligned to
rogii_feats.npz rows. Saves rogii_rawgr.npz.
    uv run python -u _cache_rawgr.py
"""
from __future__ import annotations
import os, sys, time
import numpy as np, pandas as pd
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from src.data import load_split, TARGET_INPUT
_ROGII_CACHE = os.environ.get("ROGII_CACHE", os.path.expanduser("~/rogii_cache"))
os.makedirs(_ROGII_CACHE, exist_ok=True)

t0 = time.time()
def log(m): print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)

D = np.load(f"{_ROGII_CACHE}/rogii_feats.npz")
HMM = D["hmm_pred"]
OFFS = np.linspace(-60, 60, 41)        # fine correlation grid around the HMM path (vs rich's 11 offsets)
REF_OFFS = np.linspace(-60, 60, 21)    # raw typewell reference curve (coarser)

wells = load_split(Path("data/raw/train"))
out = {}
def push(name, arr): out.setdefault(name, []).append(arr.astype(np.float32))

row_ptr = 0
for w in wells:
    h = w.horizontal; em = w.eval_mask.to_numpy(); kn = h[h[TARGET_INPUT].notna()]
    if em.sum() == 0 or len(kn) < 200:
        continue
    n = int(em.sum()); ei = np.where(em)[0]
    t = w.typewell
    tw_tvt = t["TVT"].to_numpy(float); tw_gr = gaussian_filter1d(t["GR"].ffill().bfill().to_numpy(float), 3)
    o = np.argsort(tw_tvt); tw_tvt, tw_gr = tw_tvt[o], tw_gr[o]
    tw_mean = float(np.nanmean(tw_gr)); tw_std = float(np.nanstd(tw_gr) + 1e-6)
    gr = h["GR"].interpolate(limit_direction="both").fillna(tw_mean).to_numpy(float)
    hmm_e = HMM[row_ptr:row_ptr + n]
    # raw multi-scale horizontal GR (normalized to the typewell scale)
    for s in (1.0, 3.0, 8.0):
        push(f"hg{int(s)}", (gaussian_filter1d(gr, s)[ei] - tw_mean) / tw_std)
    hgr = (gaussian_filter1d(gr, 3)[ei] - tw_mean) / tw_std
    # FINE difference grid: hgr - tw_gr(HMM + off), un-reduced correlation surface
    for k, off in enumerate(OFFS):
        ref = (np.interp(hmm_e + off, tw_tvt, tw_gr) - tw_mean) / tw_std
        push(f"dg{k}", hgr - ref)
    # raw typewell REFERENCE curve at coarser offsets (so the model sees the reference shape itself)
    for k, off in enumerate(REF_OFFS):
        push(f"rf{k}", (np.interp(hmm_e + off, tw_tvt, tw_gr) - tw_mean) / tw_std)
    row_ptr += n

res = {k: np.concatenate(v) for k, v in out.items()}
assert len(next(iter(res.values()))) == len(HMM), "misalign"
np.savez_compressed(f"{_ROGII_CACHE}/rogii_rawgr.npz", **res)
log(f"saved rogii_rawgr.npz: {len(res)} channels ({len(OFFS)} diff + {len(REF_OFFS)} ref + 3 hg), {len(HMM)} rows")
