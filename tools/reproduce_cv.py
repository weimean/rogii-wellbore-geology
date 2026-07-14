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

    # 3) OOF stack correction — LightGBM+CatBoost on step=5 subsampled rows (production config),
    #    0.6-blend with fusion. Subsampling regularizes; two boosters add diversity (notes/10).
    log("OOF stack (LGB+Cat seed-ens + MLP, step=5)...")
    import lightgbm as lgb
    from catboost import CatBoostRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPRegressor
    step = 5
    idx_in_well = np.concatenate([np.arange((g == i).sum()) for i in range(g.max() + 1)])
    sub = (idx_in_well % step) == 0
    oof_gbm = np.zeros(len(y)); oof_mlp = np.zeros(len(y))
    for tr, va in gkf.split(X, groups=g):
        ti = tr[sub[tr]]
        ps = []
        for sd in (0, 1, 2):   # seed-ensemble (matches production fit_gbm_stack)
            ps.append(lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=63,
                      min_child_samples=200, reg_lambda=1.0, subsample=0.8, subsample_freq=1,
                      colsample_bytree=0.8, random_state=sd, n_jobs=-1, verbose=-1).fit(X[ti], y[ti]).predict(X[va]))
            ps.append(CatBoostRegressor(iterations=600, learning_rate=0.03, depth=8, l2_leaf_reg=3.0,
                      rsm=0.8, random_seed=sd, thread_count=-1, verbose=0).fit(X[ti], y[ti]).predict(X[va]))
        oof_gbm[va] = np.mean(ps, axis=0)
        Xn = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0); scaler = StandardScaler().fit(Xn[ti])
        tm = ti if len(ti) <= 200000 else np.random.default_rng(0).choice(ti, 200000, replace=False)
        oof_mlp[va] = MLPRegressor(hidden_layer_sizes=(256, 128, 64), activation="relu", alpha=1e-3,
                       batch_size=4096, learning_rate_init=1e-3, early_stopping=True, n_iter_no_change=8,
                       max_iter=120, random_state=0).fit(scaler.transform(Xn[tm]), y[tm]).predict(scaler.transform(Xn[va]))
    stack_pred = 0.7 * oof_gbm + 0.3 * oof_mlp
    gbm_stack = 0.75 * (anchor + stack_pred) + 0.25 * fusion

    print("\n=== 773-well GroupKFold CV (global pooled RMSE) ===", flush=True)
    print(f"  ffill:            {score(truth, last):.3f}", flush=True)
    print(f"  level_regression: {score(truth, anchor):.3f}", flush=True)
    print(f"  hmm_fusion:       {score(truth, fusion):.3f}", flush=True)
    print(f"  gbm_stack(LGB+Cat): {score(truth, gbm_stack):.3f}", flush=True)


if __name__ == "__main__":
    main()
