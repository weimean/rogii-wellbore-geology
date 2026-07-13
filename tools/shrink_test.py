"""Uncertainty-aware shrinkage: in the deep eval zone the signed residual is proven noise, so
shrink the model's drift/wiggle toward the flat conditional mean (anchor) as eval_frac and
model-disagreement rise. Post-hoc transform on the cached OOF prediction (no retrain):

    pred = anchor + s * (pred_full - anchor),   s = clip(1 - a*ef^p - b*u, s_min, 1)

Tune (a,b,p,s_min) on DEV wells, evaluate on the LOCKBOX (seed 42) — honest. Baseline = s==1.

    uv run python -u tools/shrink_test.py
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from tools.frontier_fast import load_stack

def rmse(y, p, m=None):
    if m is not None: y, p = y[m], p[m]
    return float(np.sqrt(np.mean((y - p) ** 2)))


def main():
    s = load_stack()
    anchor, truth, g = s["anchor"], s["truth"], s["g"]
    pred_full = np.load("/tmp/rogii_oof.npz")["pred"]
    d = np.load("/tmp/rogii_feats.npz"); pf = np.load("/tmp/rogii_pf.npz")
    ef = d["eval_frac"]
    u = np.abs(d["hmm_pred"] - pf["pf_pred"])          # model disagreement
    uz = np.clip((u - u.mean()) / (u.std() + 1e-9), 0, None)

    uniq = np.unique(g)
    rng = np.random.default_rng(42)
    lock_wells = set(rng.choice(uniq, int(len(uniq) * 0.2), replace=False).tolist())
    dev = np.array([gi not in lock_wells for gi in g]); lock = ~dev

    corr = pred_full - anchor          # the correction we may shrink
    base_dev = rmse(truth, pred_full, dev); base_lock = rmse(truth, pred_full, lock)
    print(f"  baseline (no shrink): dev={base_dev:.4f}  lock={base_lock:.4f}", flush=True)

    best = (base_dev, None)
    for p in (1.0, 1.5, 2.0):
        efp = ef ** p
        for a in np.linspace(0, 1.2, 25):
            for b in (0.0, 0.05, 0.1, 0.2, 0.35):
                for smin in (0.0, 0.2, 0.4, 0.6):
                    sh = np.clip(1 - a * efp - b * uz, smin, 1.0)
                    pred = anchor + sh * corr
                    r = rmse(truth, pred, dev)
                    if r < best[0]:
                        best = (r, (a, b, p, smin))
    a, b, p, smin = best[1]
    sh = np.clip(1 - a * (ef ** p) - b * uz, smin, 1.0)
    pred = anchor + sh * corr
    print(f"\n  best schedule: a={a:.2f} b={b:.2f} p={p} s_min={smin}", flush=True)
    print(f"  shrink applied: dev={rmse(truth, pred, dev):.4f}  lock={rmse(truth, pred, lock):.4f}", flush=True)
    print(f"  delta:          dev={rmse(truth,pred,dev)-base_dev:+.4f}  lock={rmse(truth,pred,lock)-base_lock:+.4f}", flush=True)
    # where does the gain come from? deep vs shallow on lockbox
    for lo in (0.0, 0.6):
        m = lock & (ef >= lo) & (ef < lo + (0.4 if lo else 0.6))
        print(f"    lock ef[{lo:.1f},{lo+(0.4 if lo else 0.6):.1f}): base={rmse(truth,pred_full,m):.3f} -> shrink={rmse(truth,pred,m):.3f}", flush=True)
    print(f"  mean shrink factor deep (ef>0.8): {sh[ef>0.8].mean():.3f}  shallow (ef<0.2): {sh[ef<0.2].mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
