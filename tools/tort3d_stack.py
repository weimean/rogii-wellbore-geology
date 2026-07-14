"""Does proper 3D Q-tortuosity help the FULL stack (beyond our existing 1D `tort`)?
Add [dls, tort31, tort151, dlsstd51, cumtort] to the seed-ensemble LGB+Cat @ step=5 stack;
compare current vs +tort3d on lockbox (154 held-out) and pooled 773-well CV. Also report the
corr of each tort3d feature with |residual| (mycarta claims tortuosity is their best feature).

    uv run python -u tools/tort3d_stack.py
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
    X0, truth, anchor, fusion, g = s["X"], s["truth"], s["anchor"], s["fusion"], s["g"]
    y = truth - anchor
    tt = np.load("/tmp/rogii_tort3d.npz")
    T = np.column_stack([tt[k] for k in ["dls", "tort31", "tort151", "dlsstd51", "cumtort"]]).astype(np.float32)
    Xt = np.column_stack([X0, T])
    sub = train_submask(g, s["first"], s["counts"], 5)

    # gate: corr with |residual| of the current model (if OOF cached)
    if os.path.exists("/tmp/rogii_oof.npz"):
        ares = np.abs(truth - np.load("/tmp/rogii_oof.npz")["pred"])
        print("=== corr( tort3d feature , |residual| ) ===", flush=True)
        for i, nm in enumerate(["dls", "tort31", "tort151", "dlsstd51", "cumtort"]):
            print(f"  {nm:10s} {np.corrcoef(T[:, i], ares)[0,1]:+.3f}", flush=True)

    uniq = np.unique(g); rng = np.random.default_rng(42)
    lock_wells = set(rng.choice(uniq, int(len(uniq) * 0.2), replace=False).tolist())
    dev = np.array([gi not in lock_wells for gi in g]); lock = ~dev
    di = np.where(dev & sub)[0]
    def bl(p): return 0.6 * (anchor[lock] + p) + 0.4 * fusion[lock]
    def lbpred(X):
        ps = []
        for sd in (0, 1, 2):
            ps.append(lgbm(sd).fit(X[di], y[di]).predict(X[lock]))
            ps.append(cat(sd).fit(X[di], y[di]).predict(X[lock]))
        return score(truth[lock], bl(np.mean(ps, axis=0)))
    log("lockbox current..."); cur = lbpred(X0)
    log("lockbox +tort3d..."); t3 = lbpred(Xt)
    print(f"\n=== LOCKBOX (seed-ens, step=5) ===", flush=True)
    print(f"  current:   {cur:.4f}", flush=True)
    print(f"  + tort3d:  {t3:.4f}   (delta {t3-cur:+.4f})", flush=True)

    def oof(X):
        p = np.zeros(len(y))
        for tr, va in GroupKFold(5).split(X, groups=g):
            ti = tr[sub[tr]]; ps = []
            for sd in (0, 1, 2):
                ps.append(lgbm(sd).fit(X[ti], y[ti]).predict(X[va]))
                ps.append(cat(sd).fit(X[ti], y[ti]).predict(X[va]))
            p[va] = np.mean(ps, axis=0)
        return score(truth, 0.6 * (anchor + p) + 0.4 * fusion)
    log("pooled current..."); pc0 = oof(X0)
    log("pooled +tort3d..."); pct = oof(Xt)
    print(f"\n=== pooled 773-well CV ===", flush=True)
    print(f"  current:   {pc0:.4f}", flush=True)
    print(f"  + tort3d:  {pct:.4f}   (delta {pct-pc0:+.4f})", flush=True)


if __name__ == "__main__":
    main()
