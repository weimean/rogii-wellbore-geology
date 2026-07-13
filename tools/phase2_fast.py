"""Validate train-subsampling faithfulness, then run the Phase-2 meta-stack fast.

Step 1: HistGBM pooled CV at full / step=5 / step=10 train-subsampling (+ timings). If CV
matches within a few thousandths, all future iteration uses the subsample.
Step 2: HistGBM vs LightGBM vs Ridge-meta at the chosen step.

    uv run python -u tools/phase2_fast.py
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
import lightgbm as lgb
from tools.frontier_fast import load_stack, train_submask, oof
from src.validation import score

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)


def hist():
    return HistGradientBoostingRegressor(max_iter=600, learning_rate=0.03, max_leaf_nodes=63,
                                         min_samples_leaf=200, l2_regularization=1.0, random_state=0)

def lgbm():
    return lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=63,
                             min_child_samples=200, reg_lambda=1.0, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.8, random_state=0, n_jobs=-1, verbose=-1)


def main():
    log("loading stack (builds+caches on first call)...")
    s = load_stack()
    X, truth, anchor, fusion, g = s["X"], s["truth"], s["anchor"], s["fusion"], s["g"]
    y = truth - anchor
    log(f"{len(y)} rows, X{X.shape}")

    def blended(p): return 0.6 * (anchor + p) + 0.4 * fusion

    print("\n=== subsample faithfulness (HistGBM pooled CV) ===", flush=True)
    for step in [1, 5, 10]:
        tm = None if step == 1 else train_submask(g, s["first"], s["counts"], step)
        t = time.time()
        p = oof(hist, X, y, g, train_mask=tm)
        dt = time.time() - t
        n = len(y) if tm is None else int(tm.sum())
        print(f"  step={step:2d} (train rows/fold~{n//5//1000}k): CV={score(truth, blended(p)):.4f}  [{dt:5.1f}s]", flush=True)

    STEP = 5
    tm = train_submask(g, s["first"], s["counts"], STEP)
    log(f"phase-2 comparison at step={STEP}...")
    ph = oof(hist, X, y, g, train_mask=tm)
    pl = oof(lgbm, X, y, g, train_mask=tm)
    print("\n=== Phase-2 (step=5 train subsample) pooled CV ===", flush=True)
    print(f"  HistGBM:            {score(truth, blended(ph)):.4f}", flush=True)
    print(f"  LightGBM:           {score(truth, blended(pl)):.4f}", flush=True)
    print(f"  mean(Hist,LGB):     {score(truth, blended((ph+pl)/2)):.4f}", flush=True)
    fus_res = fusion - anchor
    base = np.column_stack([ph, pl, fus_res])
    meta = np.zeros(len(y))
    for tr, va in GroupKFold(5).split(base, groups=g):
        m = Ridge(1.0, positive=True).fit(base[tr], y[tr]); meta[va] = m.predict(base[va])
    mfull = Ridge(1.0, positive=True).fit(base, y)
    print(f"  Ridge-meta(H,L,fus):{score(truth, anchor + meta):.4f}  weights={np.round(mfull.coef_,3)}", flush=True)


if __name__ == "__main__":
    main()
