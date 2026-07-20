"""Reproduce the 7.016 notebook's cheap, test-legal feature families (aligned to /tmp/rogii_feats.npz):
  - multi-offset GR-match grids: hgr - typewell_GR(predictor + offset), around last_tvt (tda),
    PF (tdpf), HMM (tdhmm) -- the key novel family (a GR-match profile around each predictor).
  - finer multi-scale GR roll mean/std (grm/grs @5/21/51/101), lag/lead (glag/glead @1/5/15/30),
    gr_d1/d2/env/nrg, gr_vs_tw_anc.
Saves /tmp/rogii_rich.npz. Run: uv run python -u _cache_richfeat.py
"""
from __future__ import annotations
import os, sys, time
import numpy as np, pandas as pd
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from src.data import load_split, TARGET_INPUT
import os as _os
_ROGII_CACHE = _os.environ.get("ROGII_CACHE", _os.path.expanduser("~/rogii_cache"))
_os.makedirs(_ROGII_CACHE, exist_ok=True)

t0 = time.time()
def log(m): print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)

D = np.load(f"{_ROGII_CACHE}/rogii_feats.npz"); PF = np.load(f"{_ROGII_CACHE}/rogii_pf.npz")
HMM = D["hmm_pred"]; PFp = PF["pf_pred"]; GID = D["gidx"]
ANCH_OFFS = [-80, -40, -20, -10, -5, 0, 5, 10, 20, 40, 80]
PF_OFFS = [-30, -15, -8, -4, -2, 0, 2, 4, 8, 15, 30]

wells = load_split(Path("data/raw/train"))
cols = None; out = {}
def push(name, arr):
    out.setdefault(name, []).append(arr.astype(np.float32))

row_ptr = 0
for w in wells:
    h = w.horizontal; em = w.eval_mask.to_numpy(); kn = h[h[TARGET_INPUT].notna()]
    if em.sum() == 0 or len(kn) < 200:
        continue
    n = int(em.sum()); ei = np.where(em)[0]
    t = w.typewell
    tw_tvt = t["TVT"].to_numpy(float); tw_gr = gaussian_filter1d(t["GR"].ffill().bfill().to_numpy(float), 3)
    o = np.argsort(tw_tvt); tw_tvt, tw_gr = tw_tvt[o], tw_gr[o]
    gr = h["GR"].interpolate(limit_direction="both").fillna(np.nanmean(tw_gr)).to_numpy(float)
    gr_s = pd.Series(gaussian_filter1d(gr, 3))
    hgr = gr_s.to_numpy()[ei]
    last_tvt = float(kn[TARGET_INPUT].values[-1])
    hmm_e = HMM[row_ptr:row_ptr + n]; pf_e = PFp[row_ptr:row_ptr + n]

    # multi-offset GR-match grids
    for off in ANCH_OFFS:
        push(f"tda{off}", hgr - np.interp(last_tvt + off, tw_tvt, tw_gr))
    for off in PF_OFFS:
        push(f"tdpf{off}", hgr - np.interp(pf_e + off, tw_tvt, tw_gr))
        push(f"tdhmm{off}", hgr - np.interp(hmm_e + off, tw_tvt, tw_gr))
    push("gr_vs_tw_anc", hgr - np.interp(last_tvt, tw_tvt, tw_gr))
    # multi-scale roll + lag/lead + derivatives
    for wsz in [5, 21, 51, 101]:
        r = gr_s.rolling(wsz, center=True, min_periods=1)
        push(f"grm{wsz}", r.mean().to_numpy()[ei]); push(f"grs{wsz}", r.std().fillna(0).to_numpy()[ei])
    for lag in [1, 5, 15, 30]:
        push(f"glag{lag}", gr_s.shift(lag).bfill().to_numpy()[ei])
        push(f"glead{lag}", gr_s.shift(-lag).ffill().to_numpy()[ei])
    push("gr_d1", gr_s.diff().fillna(0).to_numpy()[ei]); push("gr_d2", gr_s.diff().diff().fillna(0).to_numpy()[ei])
    push("gr_env", gr_s.rolling(21, center=True, min_periods=1).max().to_numpy()[ei])
    push("gr_nrg", np.sqrt(np.maximum((gr_s**2).rolling(21, center=True, min_periods=1).mean(), 0)).to_numpy()[ei])
    row_ptr += n

res = {k: np.concatenate(v) for k, v in out.items()}
assert len(next(iter(res.values()))) == len(HMM), "misalign"
np.savez_compressed(f"{_ROGII_CACHE}/rogii_rich.npz", **res)
log(f"saved /tmp/rogii_rich.npz: {len(res)} features, {len(HMM)} rows")
