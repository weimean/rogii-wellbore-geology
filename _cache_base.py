"""Rebuild the base frontier caches (wiped from /tmp on reboot):
  /tmp/rogii_feats.npz  — gidx, truth, last_tvt, hmm_pred, hmm_std, eval_frac,
                          v50, v200, gr_off, known_range, eval_len, gidx
  /tmp/rogii_pf.npz     — pf_pred, pf_std

Every column comes out of one _stack_features call per well (HMM + PF each run once,
which dominates the ~10 min runtime). None of the cached columns depend on the anchor,
so we pass a zero anchor_override and extract columns by their STACK_FEATS index.
Live progress. Run foreground:  uv run python -u _cache_base.py
"""
from __future__ import annotations
import os, sys, time
import numpy as np
from pathlib import Path
from src.data import load_split, TARGET_INPUT
from src.model import _stack_features, STACK_FEATS

t0 = time.time()
def log(m): print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)

# column indices inside the X matrix _stack_features returns:
#   [STACK_FEATS...] + [anchor, hp-anchor, fusion, wt] + [pp, ps, pp-anchor, hp-pp, pf_fus]
# Save EVERY STACK_FEATS column (the Harness reads all 17) + truth/last_tvt/gidx.
PF_BASE = len(STACK_FEATS) + 4  # pp at PF_BASE, ps at PF_BASE+1

wells = load_split(Path("data/raw/train"))
log(f"loaded {len(wells)} wells")

feats = {k: [] for k in ["gidx", "truth", "last_tvt"] + STACK_FEATS}
pf = {"pf_pred": [], "pf_std": []}

gi = 0
for k, w in enumerate(wells):
    em = w.eval_mask.to_numpy()
    if em.sum() == 0:
        continue
    n = int(em.sum())
    f = _stack_features(w, None, anchor_override=np.zeros(n))
    if f is None or "truth" not in f:
        continue
    X = f["X"]
    feats["gidx"].append(np.full(n, gi))
    feats["truth"].append(f["truth"])
    feats["last_tvt"].append(f["ff"][em])
    for ci, name in enumerate(STACK_FEATS):
        feats[name].append(X[:, ci])
    pf["pf_pred"].append(X[:, PF_BASE])
    pf["pf_std"].append(X[:, PF_BASE + 1])
    gi += 1
    if (k + 1) % 150 == 0:
        log(f"  {k+1}/{len(wells)}  ({gi} kept)")

feats = {k: np.concatenate(v) for k, v in feats.items()}
pf = {k: np.concatenate(v) for k, v in pf.items()}
assert len(feats["truth"]) == len(pf["pf_pred"])
np.savez_compressed("/tmp/rogii_feats.npz", **feats)
np.savez_compressed("/tmp/rogii_pf.npz", **pf)
log(f"saved: {len(feats['truth'])} rows, {gi} wells → /tmp/rogii_feats.npz + /tmp/rogii_pf.npz")
