"""Verify the MLP-diversity gain on POOLED 773-well CV (not just one lockbox fold). OOF LGB+Cat
(seed-ens) + OOF MLP, blend, pooled RMSE. Rich features. w=0.75.

    uv run python -u tools/mlp_pooled.py
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
import lightgbm as lgb
from catboost import CatBoostRegressor
from tools.frontier_fast import load_stack, train_submask
from src.validation import score

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)


def main():
    s = load_stack()
    X0, truth, anchor, fusion, g = s["X"], s["truth"], s["anchor"], s["fusion"], s["g"]
    y = truth - anchor
    r = np.load("/tmp/rogii_rich.npz")
    X = np.column_stack([X0] + [r[k] for k in r.files]).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    sub = train_submask(g, s["first"], s["counts"], 5)
    gp = np.zeros(len(y)); mp = np.zeros(len(y))
    for fi, (tr, va) in enumerate(GroupKFold(5).split(X, groups=g)):
        ti = tr[sub[tr]]
        for sd in (0, 1, 2):
            gp[va] += lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=63, min_child_samples=200,
                        reg_lambda=1.0, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=sd,
                        n_jobs=-1, verbose=-1).fit(X[ti], y[ti]).predict(X[va])
            gp[va] += CatBoostRegressor(iterations=600, learning_rate=0.03, depth=8, l2_leaf_reg=3.0, rsm=0.8,
                        random_seed=sd, thread_count=-1, verbose=0).fit(X[ti], y[ti]).predict(X[va])
        tm = ti if len(ti) <= 200000 else np.random.default_rng(0).choice(ti, 200000, replace=False)
        mp[va] = make_pipeline(StandardScaler(),
                    MLPRegressor(hidden_layer_sizes=(256, 128, 64), activation="relu", alpha=1e-3,
                        batch_size=4096, learning_rate_init=1e-3, early_stopping=True, n_iter_no_change=8,
                        max_iter=120, random_state=0)).fit(X[tm], y[tm]).predict(X[va])
        log(f"fold {fi} done")
    gp /= 6
    def sc(p): return score(truth, 0.75 * (anchor + p) + 0.25 * fusion)
    print("\n=== pooled 773-well CV (w=0.75) ===", flush=True)
    print(f"  LGB+Cat:           {sc(gp):.4f}", flush=True)
    print(f"  MLP alone:         {sc(mp):.4f}", flush=True)
    for wm in (0.2, 0.3, 0.4):
        print(f"  {(1-wm):.1f}*GBM+{wm:.1f}*MLP:  {sc((1-wm)*gp + wm*mp):.4f}", flush=True)
    print(f"  corr(GBM,MLP)={np.corrcoef(gp,mp)[0,1]:.3f}", flush=True)


if __name__ == "__main__":
    main()
