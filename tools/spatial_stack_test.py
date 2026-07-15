"""Incremental pooled-CV delta of the spatial-formation features over the rich stack.
LGB+Cat seed-ens (no MLP, for speed), w=0.75, step=5. Compares base+rich vs base+rich+spatial.

    uv run python -u tools/spatial_stack_test.py
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
    Xr = np.column_stack([X0] + [r[k] for k in r.files]).astype(np.float32)
    Xr = np.nan_to_num(Xr, nan=0.0, posinf=0.0, neginf=0.0)
    sp = np.load("/tmp/rogii_spatial_feat.npz")
    Xsp = np.column_stack([Xr] + [sp[k] for k in sp.files]).astype(np.float32)
    Xsp = np.nan_to_num(Xsp, nan=0.0, posinf=0.0, neginf=0.0)
    sub = train_submask(g, s["first"], s["counts"], 5)
    log(f"rich {Xr.shape[1]} feats -> +spatial {Xsp.shape[1]} feats")

    base = oof(Xr, y, g, sub, anchor, fusion, truth); log(f"base (rich):        {base:.4f}")
    withsp = oof(Xsp, y, g, sub, anchor, fusion, truth); log(f"+spatial features:  {withsp:.4f}")
    print(f"\n=== pooled 773-well CV (w=0.75, LGB+Cat seed-ens) ===", flush=True)
    print(f"  rich:            {base:.4f}", flush=True)
    print(f"  rich + spatial:  {withsp:.4f}   delta {base - withsp:+.4f}", flush=True)


if __name__ == "__main__":
    main()
