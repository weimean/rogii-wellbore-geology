"""Phase 1: tune the HMM by coordinate descent on a held-out well split.

Honest protocol (HMM is per-well, no cross-well fitting):
  tune  = wells[:150]   -> pick params here
  hold  = wells[150:300] -> report generalization
  lm_wells = wells[300:] -> fit the level anchor (disjoint from tune+hold)
Metric = FUSED RMSE (HMM + level_regression by posterior variance), best blend_k free.
Prints progress live (run with `python -u`).
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from pathlib import Path
from src.data import load_split
from src.model import fit_level_model, level_regression, hmm_posterior
from src.validation import score

t0 = time.time()
def log(m): print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)

wells = load_split(Path("data/raw/train"))
tune_w, hold_w, lm_w = wells[:150], wells[150:300], wells[300:]
log(f"loaded {len(wells)} wells; fitting level model on {len(lm_w)}...")
lm = fit_level_model(lm_w)

def prep(ws, tag):
    out = []
    for i, w in enumerate(ws):
        em = w.eval_mask.to_numpy()
        if em.sum() == 0:
            continue
        out.append((w, em, w.horizontal[w.eval_mask]["TVT"].values, level_regression(w, lm)[em]))
    log(f"prepped {tag}: {len(out)} wells")
    return out

tune, hold = prep(tune_w, "tune"), prep(hold_w, "hold")
BKS = [2.0, 3.0, 4.0, 6.0]

def evaluate(wset, params):
    T, preds = [], {bk: [] for bk in BKS}
    for w, em, truth, anchor in wset:
        hp, hs = hmm_posterior(w, **params)
        T.append(truth)
        for bk in BKS:
            wt = 1.0 / (1.0 + (hs[em] / bk) ** 2)
            preds[bk].append(wt * hp[em] + (1 - wt) * anchor)
    T = np.concatenate(T)
    return min((score(T, np.concatenate(preds[bk])), bk) for bk in BKS)

DEFAULTS = dict(mom=0.98, sig_p=0.45, rate_span=0.035, lam=1.0, band_pad=90.0, sig_r=0.003)
GRID = dict(mom=[0.96, 0.98, 0.99, 0.995], sig_p=[0.3, 0.45, 0.6, 0.8],
            rate_span=[0.025, 0.035, 0.05, 0.07], lam=[0.6, 0.8, 1.0],
            band_pad=[60.0, 90.0, 120.0], sig_r=[0.002, 0.003, 0.005])

cur = dict(DEFAULTS)
base, base_bk = evaluate(tune, cur)
log(f"start: tune={base:.3f} bk={base_bk}  (defaults)")
for p in ["mom", "sig_p", "rate_span", "lam", "band_pad", "sig_r"]:
    best = (base, cur[p], base_bk)
    for v in GRID[p]:
        if v == cur[p]:
            continue
        r, bk = evaluate(tune, {**cur, p: v})
        log(f"  {p}={v}: tune={r:.3f} (bk={bk})")
        if r < best[0]:
            best = (r, v, bk)
    cur[p], base = best[1], best[0]
    log(f">> {p} = {best[1]}  (tune {best[0]:.3f})")

log("evaluating on HOLDOUT...")
hr, hbk = evaluate(hold, cur)
dr, dbk = evaluate(hold, DEFAULTS)
print("\n==================== RESULT ====================", flush=True)
print(f"tuned params : {cur}", flush=True)
print(f"HOLDOUT tuned    : {hr:.3f} (bk={hbk})", flush=True)
print(f"HOLDOUT defaults : {dr:.3f} (bk={dbk})", flush=True)
print(f"holdout delta    : {hr - dr:+.3f}", flush=True)
