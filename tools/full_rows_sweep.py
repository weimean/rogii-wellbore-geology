"""Training-regime sweep (notes/26): our step=5 subsample-as-regularization vs the sub-9 teams' full-rows
+ strong-regularization regime. CV-HONEST (a model-regularization knob, not a blend — grouped CV tells the
truth here, unlike blend weights). Same base (rich+formation ~10.92), fixed blend w=0.75, single seed for
speed (we compare REGIMES, not final seed-ensembled performance). If a regime clears grouped CV it's worth
a submission.

    uv run python -u tools/full_rows_sweep.py
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

# (name, step, lgb_kwargs, cat_kwargs)
CONFIGS = [
    ("cur_s5_reg1",  5, dict(num_leaves=63,  min_child_samples=200, reg_lambda=1.0, n_estimators=600),
                        dict(depth=8, l2_leaf_reg=3.0,  iterations=600)),
    ("s5_reg5",      5, dict(num_leaves=63,  min_child_samples=200, reg_lambda=5.0, n_estimators=600),
                        dict(depth=8, l2_leaf_reg=10.0, iterations=600)),
    ("s2_reg5_big",  2, dict(num_leaves=127, min_child_samples=80,  reg_lambda=5.0, n_estimators=900),
                        dict(depth=8, l2_leaf_reg=10.0, iterations=900)),
    ("full_reg_big", 1, dict(num_leaves=255, min_child_samples=20,  reg_lambda=5.0, n_estimators=1200),
                        dict(depth=8, l2_leaf_reg=10.0, iterations=1200)),
]


def oof(X, y, g, sub, anchor, fusion, truth, lk, ck):
    gp = np.zeros(len(y))
    for tr, va in GroupKFold(5).split(X, groups=g):
        ti = tr[sub[tr]]
        gp[va] += lgb.LGBMRegressor(learning_rate=0.03, subsample=0.8, subsample_freq=1,
                    colsample_bytree=0.8, random_state=0, n_jobs=-1, verbose=-1, **lk).fit(X[ti], y[ti]).predict(X[va])
        gp[va] += CatBoostRegressor(learning_rate=0.03, rsm=0.8, random_seed=0, thread_count=-1,
                    verbose=0, **ck).fit(X[ti], y[ti]).predict(X[va])
    gp /= 2
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
    log(f"base {Xbase.shape[1]} feats, {len(y)} rows")

    print("\n=== training-regime sweep (base rich+formation, w=0.75, single seed) ===", flush=True)
    for name, step, lk, ck in CONFIGS:
        sub = train_submask(g, s["first"], s["counts"], step)
        n_tr = int(sub.sum())
        t1 = time.time()
        rmse = oof(Xbase, y, g, sub, anchor, fusion, truth, lk, ck)
        log(f"  {name:14s} step={step} ({n_tr} train rows): {rmse:.4f}   [{time.time()-t1:.0f}s]")


if __name__ == "__main__":
    main()
