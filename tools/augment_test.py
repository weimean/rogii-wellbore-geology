"""Does prefix-cut AUGMENTATION improve GBM generalization? Train the seed-ensemble LGB+Cat on
base (real eval, step=5) rows + synthetic-PS augmentation rows (from the same DEV wells only),
predict the real eval zone. Measure lockbox (held-out) + pooled. If the lockbox improves, it's a
real generalization gain (not LB-chasing). blend w=0.75.

    uv run python -u tools/augment_test.py
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

def lgbm(sd):
    return lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=63,
                             min_child_samples=200, reg_lambda=1.0, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.8, random_state=sd, n_jobs=-1, verbose=-1)
def cat(sd):
    return CatBoostRegressor(iterations=600, learning_rate=0.03, depth=8, l2_leaf_reg=3.0,
                             rsm=0.8, random_seed=sd, thread_count=-1, verbose=0)


def main():
    s = load_stack()
    X, truth, anchor, fusion, g = s["X"], s["truth"], s["anchor"], s["fusion"], s["g"]
    y = truth - anchor
    sub = train_submask(g, s["first"], s["counts"], 5)
    aug = np.load("/tmp/rogii_augment.npz")
    Xa, ya, ga = aug["X_aug"].astype(X.dtype), aug["y_aug"], aug["g_aug"]
    log(f"base rows {len(y)}, augmentation rows {len(ya)} from {len(np.unique(ga))} wells")

    uniq = np.unique(g); rng = np.random.default_rng(42)
    lock_wells = set(rng.choice(uniq, int(len(uniq) * 0.2), replace=False).tolist())
    dev = np.array([gi not in lock_wells for gi in g]); lock = ~dev
    dev_aug = np.array([gi not in lock_wells for gi in ga])
    di = np.where(dev & sub)[0]; ai = np.where(dev_aug)[0]
    def bl(p): return 0.75 * (anchor[lock] + p) + 0.25 * fusion[lock]

    def fit_pred(with_aug):
        Xt = np.vstack([X[di], Xa[ai]]) if with_aug else X[di]
        yt = np.concatenate([y[di], ya[ai]]) if with_aug else y[di]
        ps = []
        for sd in (0, 1, 2):
            ps.append(lgbm(sd).fit(Xt, yt).predict(X[lock]))
            ps.append(cat(sd).fit(Xt, yt).predict(X[lock]))
        return score(truth[lock], bl(np.mean(ps, axis=0)))

    log("lockbox: base only..."); base = fit_pred(False)
    log("lockbox: + augmentation..."); aug_s = fit_pred(True)
    print("\n=== LOCKBOX (seed-ens, step=5, w=0.75) ===", flush=True)
    print(f"  base only      : {base:.4f}", flush=True)
    print(f"  + augmentation : {aug_s:.4f}   (delta {aug_s-base:+.4f})", flush=True)


if __name__ == "__main__":
    main()
