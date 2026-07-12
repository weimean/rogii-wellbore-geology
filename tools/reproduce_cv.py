"""Reproduce the headline 773-well GroupKFold CV table for every production predictor.

    uv run python tools/reproduce_cv.py

HMM posteriors are per-well (no cross-well leakage) so they're computed once and reused;
level_regression and the GBM stack are fit out-of-fold. Runtime ~8 min (the HMM dominates).
Expected: ffill 15.91 · level 15.33 · hmm_fusion 14.24 · gbm_stack 13.77.
"""
from __future__ import annotations
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingRegressor
from src.data import load_split, TARGET_INPUT
from src.model import (fit_level_model, level_regression, hmm_posterior, baseline_ffill,
                       _stack_features)
from src.validation import score

TRAIN = Path("data/raw/train")


def main():
    wells = load_split(TRAIN)
    print(f"loaded {len(wells)} wells; computing per-well features (HMM dominates ~8min)...")

    feats, per_well = [], []      # feats: stack feature dict per well; per_well: (name, truth, anchor_slot)
    groups, truth_all = [], []
    # level model needs OOF; first gather level-feature table, fit OOF, then build stack rows
    names = [w.name for w in wells]
    # OOF level anchor
    gkf = GroupKFold(5)
    oof_level = {}
    for tr, va in gkf.split(wells, groups=np.array(names)):
        lm = fit_level_model([wells[i] for i in tr])
        for i in va:
            w = wells[i]; em = w.eval_mask.to_numpy()
            if em.sum() == 0:
                continue
            oof_level[w.name] = level_regression(w, lm)[em]

    # full-data level model (for the stack feature 'anchor' we use OOF; recompute features per well)
    lm_full = fit_level_model(wells)
    rows = []
    for k, w in enumerate(wells):
        f = _stack_features(w, lm_full)
        if f is None or "truth" not in f:
            continue
        name = w.name
        anchor = oof_level.get(name)
        if anchor is None:
            continue
        # swap the in-sample anchor column for the OOF anchor so CV is honest
        # stack X columns end with [anchor, hp-anchor, fusion, wt]; rebuild those 4 from OOF anchor
        hp = f["X"][:, 0]                     # hmm_pred is first STACK_FEAT
        hs = f["X"][:, 1]                     # hmm_std is second
        wt = 1.0 / (1.0 + (hs / 4.0) ** 2)
        fusion = wt * hp + (1 - wt) * anchor
        X = np.column_stack([f["X"][:, :len(f["X"][0]) - 4], anchor, hp - anchor, fusion, wt])
        rows.append(dict(name=name, X=X, anchor=anchor, fusion=fusion, hp=hp,
                         truth=f["truth"], last=baseline_ffill(w.horizontal)[w.eval_mask.to_numpy()]))
        if (k + 1) % 150 == 0:
            print(f"  {k+1}/{len(wells)}")

    g = np.concatenate([np.full(len(r["truth"]), i) for i, r in enumerate(rows)])
    truth = np.concatenate([r["truth"] for r in rows])
    last = np.concatenate([r["last"] for r in rows])
    anchor = np.concatenate([r["anchor"] for r in rows])
    fusion = np.concatenate([r["fusion"] for r in rows])
    X = np.vstack([r["X"] for r in rows])
    y = truth - anchor

    # OOF GBM correction
    oof_gbm = np.zeros(len(y))
    for tr, va in gkf.split(X, groups=g):
        m = HistGradientBoostingRegressor(max_iter=600, learning_rate=0.03, max_leaf_nodes=63,
                                          min_samples_leaf=200, l2_regularization=1.0, random_state=0)
        m.fit(X[tr], y[tr]); oof_gbm[va] = m.predict(X[va])
    gbm_stack = 0.5 * (anchor + oof_gbm) + 0.5 * fusion

    print("\n=== 773-well GroupKFold CV (global pooled RMSE) ===")
    print(f"  ffill:            {score(truth, last):.3f}")
    print(f"  level_regression: {score(truth, anchor):.3f}")
    print(f"  hmm_fusion:       {score(truth, fusion):.3f}")
    print(f"  gbm_stack:        {score(truth, gbm_stack):.3f}")


if __name__ == "__main__":
    main()
