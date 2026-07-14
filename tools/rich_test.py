"""Track A: do the 7.016 notebook's cheap features (multi-offset GR-match grids + finer GR)
help OUR LGB+Cat stack? Add /tmp/rogii_rich.npz (54 feats) to the stack; lockbox + pooled, w=0.75.

    uv run python -u tools/rich_test.py
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
    return lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=63, min_child_samples=200,
                             reg_lambda=1.0, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                             random_state=sd, n_jobs=-1, verbose=-1)
def cat(sd):
    return CatBoostRegressor(iterations=600, learning_rate=0.03, depth=8, l2_leaf_reg=3.0, rsm=0.8,
                             random_seed=sd, thread_count=-1, verbose=0)


def main():
    s = load_stack()
    X0, truth, anchor, fusion, g = s["X"], s["truth"], s["anchor"], s["fusion"], s["g"]
    y = truth - anchor
    r = np.load("/tmp/rogii_rich.npz")
    R = np.column_stack([r[k] for k in r.files]).astype(np.float32)
    Xr = np.column_stack([X0, R])
    log(f"base {X0.shape[1]} feats + rich {R.shape[1]} = {Xr.shape[1]}")
    sub = train_submask(g, s["first"], s["counts"], 5)
    uniq = np.unique(g); rng = np.random.default_rng(42)
    lock_wells = set(rng.choice(uniq, int(len(uniq) * 0.2), replace=False).tolist())
    dev = np.array([gi not in lock_wells for gi in g]); lock = ~dev
    di = np.where(dev & sub)[0]
    def bl(p): return 0.75 * (anchor[lock] + p) + 0.25 * fusion[lock]
    def lbp(X):
        ps = []
        for sd in (0, 1, 2):
            ps.append(lgbm(sd).fit(X[di], y[di]).predict(X[lock]))
            ps.append(cat(sd).fit(X[di], y[di]).predict(X[lock]))
        return score(truth[lock], bl(np.mean(ps, axis=0)))
    log("lockbox base..."); b = lbp(X0)
    log("lockbox +rich..."); rr = lbp(Xr)
    print(f"\n=== LOCKBOX (seed-ens, w=0.75) ===", flush=True)
    print(f"  base (37 feats):  {b:.4f}", flush=True)
    print(f"  + rich (91 feats): {rr:.4f}   (delta {rr-b:+.4f})", flush=True)
    if rr < b - 0.02:
        def oof(X):
            p = np.zeros(len(y))
            for tr, va in GroupKFold(5).split(X, groups=g):
                ti = tr[sub[tr]]; ps = []
                for sd in (0, 1, 2):
                    ps.append(lgbm(sd).fit(X[ti], y[ti]).predict(X[va]))
                    ps.append(cat(sd).fit(X[ti], y[ti]).predict(X[va]))
                p[va] = np.mean(ps, axis=0)
            return score(truth, 0.75 * (anchor + p) + 0.25 * fusion)
        log("pooled base..."); pb = oof(X0); log("pooled +rich..."); pr = oof(Xr)
        print(f"\n=== pooled 773-well CV ===\n  base {pb:.4f}\n  +rich {pr:.4f}  (delta {pr-pb:+.4f})", flush=True)


if __name__ == "__main__":
    main()
