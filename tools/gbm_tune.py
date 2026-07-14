"""Whole-pipeline optics #2 — focused GBM hyperparameter check on the lockbox (seed-ensemble,
step=5, blend 0.70). A few targeted variations vs current; adopt only if it robustly beats
current on the lockbox (we expect little — 'model not the bottleneck'). Single dev-fit each.

    uv run python -u tools/gbm_tune.py
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import lightgbm as lgb
from catboost import CatBoostRegressor
from tools.frontier_fast import load_stack, train_submask
from src.validation import score

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)


def main():
    s = load_stack()
    X, truth, anchor, fusion, g = s["X"], s["truth"], s["anchor"], s["fusion"], s["g"]
    y = truth - anchor
    sub = train_submask(g, s["first"], s["counts"], 5)
    uniq = np.unique(g); rng = np.random.default_rng(42)
    lock_wells = set(rng.choice(uniq, int(len(uniq) * 0.2), replace=False).tolist())
    dev = np.array([gi not in lock_wells for gi in g]); lock = ~dev
    di = np.where(dev & sub)[0]
    def bl(p): return 0.70 * (anchor[lock] + p) + 0.30 * fusion[lock]

    def run(lgb_kw, cat_kw):
        ps = []
        for sd in (0, 1, 2):
            ps.append(lgb.LGBMRegressor(random_state=sd, n_jobs=-1, verbose=-1, **lgb_kw).fit(X[di], y[di]).predict(X[lock]))
            ps.append(CatBoostRegressor(random_seed=sd, thread_count=-1, verbose=0, **cat_kw).fit(X[di], y[di]).predict(X[lock]))
        return score(truth[lock], bl(np.mean(ps, axis=0)))

    L0 = dict(n_estimators=600, learning_rate=0.03, num_leaves=63, min_child_samples=200,
              reg_lambda=1.0, subsample=0.8, subsample_freq=1, colsample_bytree=0.8)
    C0 = dict(iterations=600, learning_rate=0.03, depth=8, l2_leaf_reg=3.0, rsm=0.8)

    configs = {
        "current": (L0, C0),
        "deeper (leaves127,depth10)": ({**L0, "num_leaves": 127}, {**C0, "depth": 10}),
        "shallower (leaves31,depth6)": ({**L0, "num_leaves": 31}, {**C0, "depth": 6}),
        "more_reg (min_child400,l2_10)": ({**L0, "min_child_samples": 400}, {**C0, "l2_leaf_reg": 10.0}),
        "slower (lr0.02,n900/iter900)": ({**L0, "learning_rate": 0.02, "n_estimators": 900},
                                          {**C0, "learning_rate": 0.02, "iterations": 900}),
    }
    print("\n=== LOCKBOX (seed-ens, step=5, blend 0.70) ===", flush=True)
    for name, (lk, ck) in configs.items():
        log(f"fit [{name}]...")
        print(f"  {name:32s} {run(lk, ck):.4f}", flush=True)


if __name__ == "__main__":
    main()
