"""Reproduce the headline 773-well GroupKFold CV table for every production predictor.

    uv run python -u tools/reproduce_cv.py

HMM/PF posteriors are per-well (no cross-well leakage); the level anchor is out-of-fold
(passed into _stack_features via anchor_override) so the GBM stack CV is honest. The GBM
itself is fit out-of-fold. Runtime ~10 min (HMM+PF per well dominates).
Expected: ffill 15.91 · level 15.33 · hmm_fusion 14.24 · gbm_stack ~13.53.
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingRegressor
from src.data import load_split, TARGET_INPUT
from src.model import (fit_level_model, level_regression, baseline_ffill, _stack_features)
from src.validation import score

t0 = time.time()
def log(m): print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


def main():
    wells = load_split(Path("data/raw/train"))
    names = np.array([w.name for w in wells])
    gkf = GroupKFold(5)

    # 1) OOF level anchor per well (== level_regression, out-of-fold)
    log(f"loaded {len(wells)} wells; OOF level anchor...")
    oof_anchor = {}
    for tr, va in gkf.split(wells, groups=names):
        lm = fit_level_model([wells[i] for i in tr])
        for i in va:
            w = wells[i]; em = w.eval_mask.to_numpy()
            if em.sum():
                oof_anchor[w.name] = level_regression(w, lm)[em]

    # 2) per-well stack features with the OOF anchor baked in (HMM+PF dominate runtime)
    log("building stack features (HMM + PF per well)...")
    rows = []
    lm_full = fit_level_model(wells)  # only used for the (unused-here) in-sample path guard
    for k, w in enumerate(wells):
        if w.name not in oof_anchor:
            continue
        f = _stack_features(w, lm_full, anchor_override=oof_anchor[w.name])
        if f is None or "truth" not in f:
            continue
        rows.append((k, f))
        if (k + 1) % 150 == 0:
            log(f"  {k+1}/{len(wells)}")

    g = np.concatenate([np.full(len(f["truth"]), i) for i, (_, f) in enumerate(rows)])
    truth = np.concatenate([f["truth"] for _, f in rows])
    anchor = np.concatenate([f["anchor"] for _, f in rows])
    fusion = np.concatenate([f["fusion"] for _, f in rows])
    last = np.concatenate([f["ff"][f["em"]] for _, f in rows])
    X = np.vstack([f["X"] for _, f in rows])
    y = truth - anchor

    # 3) OOF GBM correction (0.6-blend with fusion, matching production gbm_stack)
    log("OOF GBM...")
    oof_gbm = np.zeros(len(y))
    for tr, va in gkf.split(X, groups=g):
        m = HistGradientBoostingRegressor(max_iter=600, learning_rate=0.03, max_leaf_nodes=63,
                                          min_samples_leaf=200, l2_regularization=1.0, random_state=0)
        m.fit(X[tr], y[tr]); oof_gbm[va] = m.predict(X[va])
    gbm_stack = 0.6 * (anchor + oof_gbm) + 0.4 * fusion

    print("\n=== 773-well GroupKFold CV (global pooled RMSE) ===", flush=True)
    print(f"  ffill:            {score(truth, last):.3f}", flush=True)
    print(f"  level_regression: {score(truth, anchor):.3f}", flush=True)
    print(f"  hmm_fusion:       {score(truth, fusion):.3f}", flush=True)
    print(f"  gbm_stack (+PF):  {score(truth, gbm_stack):.3f}", flush=True)


if __name__ == "__main__":
    main()
