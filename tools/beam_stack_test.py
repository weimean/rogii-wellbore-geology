"""Grouped-CV pooled-RMSE delta of the BEAM-DP family (/tmp/rogii_beam.npz) on top of the full base
(base + rich + formation ~= 10.90). LGB+Cat seed-ens, w=0.75, step=5. Copies grpeak_stack_test.py.
NOTE: like all tracker/spatial features, grouped CV UNDERSTATES this — a positive here is encouraging,
a flat is ambiguous (public is the arbiter). Beam TVT paths + mean are fed anchor-relative.

    uv run python -u tools/beam_stack_test.py
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from sklearn.model_selection import GroupKFold
import lightgbm as lgb
from catboost import CatBoostRegressor
from tools.frontier_fast import load_stack, train_submask
from src.validation import score

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)
RAW_TVT = {"beam_cons", "beam_loose", "beam_vcons", "beam_sm5", "beam_vloose", "beam_mid", "beam_stiff", "beam_mean"}


def oof(X, y, g, sub, anchor, fusion, truth):
    gp = np.zeros(len(y))
    for tr, va in GroupKFold(5).split(X, groups=g):
        ti = tr[sub[tr]]
        for sd in (0, 1, 2):
            gp[va] += lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=63,
                        min_child_samples=200, reg_lambda=1.0, subsample=0.8, subsample_freq=1,
                        colsample_bytree=0.8, random_state=sd, n_jobs=-1, verbose=-1).fit(X[ti], y[ti]).predict(X[va])
            gp[va] += CatBoostRegressor(iterations=600, learning_rate=0.03, depth=8, l2_leaf_reg=3.0,
                        rsm=0.8, random_seed=sd, thread_count=-1, verbose=0).fit(X[ti], y[ti]).predict(X[va])
    gp /= 6
    return score(truth, 0.75 * (anchor + gp) + 0.25 * fusion)


def main():
    s = load_stack()
    X0, truth, anchor, fusion, g = s["X"], s["truth"], s["anchor"], s["fusion"], s["g"]
    y = truth - anchor
    r = np.load("/tmp/rogii_rich.npz")
    Xr = np.nan_to_num(np.column_stack([X0] + [r[k] for k in r.files]), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    fo = np.load("/tmp/rogii_formation.npz")
    fcols = []
    for k in fo.files:
        v = fo[k].astype(np.float32)
        if k.startswith("tvtF") or k == "tvt_dense":
            v = v - anchor.astype(np.float32)
        fcols.append(v)
    Xbase = np.nan_to_num(np.column_stack([Xr] + fcols), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    bz = np.load("/tmp/rogii_beam.npz")
    bcols = []
    for k in bz.files:
        v = bz[k].astype(np.float32)
        if k in RAW_TVT:                      # raw TVT paths -> anchor-relative
            v = v - anchor.astype(np.float32)
        bcols.append(v)
    Xb = np.nan_to_num(np.column_stack([Xbase] + bcols), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    sub = train_submask(g, s["first"], s["counts"], 5)
    log(f"base {Xbase.shape[1]} -> +beam {Xb.shape[1]} feats ({len(bz.files)} new)")

    base = oof(Xbase, y, g, sub, anchor, fusion, truth); log(f"base(rich+formation): {base:.4f}")
    withb = oof(Xb, y, g, sub, anchor, fusion, truth); log(f"+beam:                {withb:.4f}")
    print(f"\n=== pooled 773-well grouped CV (w=0.75, LGB+Cat seed-ens) ===", flush=True)
    print(f"  base (rich+formation):     {base:.4f}", flush=True)
    print(f"  + beam-DP family:          {withb:.4f}   delta {base - withb:+.4f}", flush=True)


if __name__ == "__main__":
    main()
