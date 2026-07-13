"""Does the geology LEVEL feature help the FULL stack (not just the anchor)?
Add [geo_last_off, geo_hmm_off, geo_last_med] to the LGB+Cat @ step=5 stack; compare current
vs +geo on the lockbox (154 held-out wells) and pooled 773-well CV.

    uv run python -u tools/geology_stack.py
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

def lgbm():
    return lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=63,
                             min_child_samples=200, reg_lambda=1.0, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.8, random_state=0, n_jobs=-1, verbose=-1)
def cat():
    return CatBoostRegressor(iterations=600, learning_rate=0.03, depth=8, l2_leaf_reg=3.0,
                             rsm=0.8, random_seed=0, thread_count=-1, verbose=0)


def main():
    s = load_stack()
    X0, truth, anchor, fusion, g = s["X"], s["truth"], s["anchor"], s["fusion"], s["g"]
    y = truth - anchor
    geo = np.load("/tmp/rogii_geology.npz")
    G = np.column_stack([geo["geo_last_off"], geo["geo_hmm_off"], geo["geo_last_med"]]).astype(np.float32)
    Xg = np.column_stack([X0, G])
    sub = train_submask(g, s["first"], s["counts"], 5)

    uniq = np.unique(g)
    rng = np.random.default_rng(42)
    lock_wells = set(rng.choice(uniq, int(len(uniq) * 0.2), replace=False).tolist())
    dev = np.array([gi not in lock_wells for gi in g]); lock = ~dev
    dev_sub = np.where(dev & sub)[0]

    def lb_pred(X):
        pl = lgbm().fit(X[dev_sub], y[dev_sub]).predict(X[lock])
        pc = cat().fit(X[dev_sub], y[dev_sub]).predict(X[lock])
        p = 0.5 * (pl + pc)
        return score(truth[lock], 0.6 * (anchor[lock] + p) + 0.4 * fusion[lock])

    log("lockbox: current stack...")
    cur = lb_pred(X0)
    log("lockbox: + geology...")
    geo_lb = lb_pred(Xg)
    print("\n=== LOCKBOX (154 held-out wells), LGB+Cat @ step=5 ===", flush=True)
    print(f"  current stack:  {cur:.4f}", flush=True)
    print(f"  + geology:      {geo_lb:.4f}   (delta {geo_lb-cur:+.4f})", flush=True)

    def oof(X):
        p = np.zeros(len(y))
        for tr, va in GroupKFold(5).split(X, groups=g):
            ti = tr[sub[tr]]
            pl = lgbm().fit(X[ti], y[ti]).predict(X[va])
            pc = cat().fit(X[ti], y[ti]).predict(X[va])
            p[va] = 0.5 * (pl + pc)
        return score(truth, 0.6 * (anchor + p) + 0.4 * fusion)

    log("pooled: current...")
    pc0 = oof(X0)
    log("pooled: + geology...")
    pcg = oof(Xg)
    print("\n=== pooled 773-well CV, LGB+Cat @ step=5 ===", flush=True)
    print(f"  current stack:  {pc0:.4f}", flush=True)
    print(f"  + geology:      {pcg:.4f}   (delta {pcg-pc0:+.4f})", flush=True)


if __name__ == "__main__":
    main()
