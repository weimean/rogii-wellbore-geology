"""Lockbox gate for the Phase-2 win: does mean(HistGBM,LightGBM) at step=5 beat the current
HistGBM on HELD-OUT wells (154 never-tuned), not just pooled CV?

Split dev(80%)/lockbox(20%) by well (seed 42, same as tools/frontier_cv.py). Fit on dev,
predict lockbox. Report current (HistGBM full) vs candidate (mean(H,L), step=5).

    uv run python -u tools/phase2_lockbox.py
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
import lightgbm as lgb
from tools.frontier_fast import load_stack, train_submask
from src.validation import score

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)


def hist():
    return HistGradientBoostingRegressor(max_iter=600, learning_rate=0.03, max_leaf_nodes=63,
                                         min_samples_leaf=200, l2_regularization=1.0, random_state=0)

def lgbm():
    return lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=63,
                             min_child_samples=200, reg_lambda=1.0, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.8, random_state=0, n_jobs=-1, verbose=-1)


def main():
    s = load_stack()
    X, truth, anchor, fusion, g = s["X"], s["truth"], s["anchor"], s["fusion"], s["g"]
    y = truth - anchor
    uniq = np.unique(g)
    rng = np.random.default_rng(42)
    lock_wells = set(rng.choice(uniq, int(len(uniq) * 0.2), replace=False).tolist())
    dev = np.array([gi not in lock_wells for gi in g]); lock = ~dev
    ef = np.repeat(np.arange(len(uniq)), s["counts"])  # placeholder; eval_frac not needed here
    log(f"dev rows={dev.sum()} lock rows={lock.sum()} (lock wells={len(lock_wells)})")

    def blended(p): return 0.6 * (anchor[lock] + p) + 0.4 * fusion[lock]

    sub = train_submask(g, s["first"], s["counts"], 5)
    dev_full = np.where(dev)[0]
    dev_sub = np.where(dev & sub)[0]

    def fit_pred(make, rows):
        m = make(); m.fit(X[rows], y[rows]); return m.predict(X[lock])

    log("current: HistGBM full dev...")
    cur = fit_pred(hist, dev_full)
    log("candidate: HistGBM + LightGBM, step=5 dev...")
    ph = fit_pred(hist, dev_sub)
    pl = fit_pred(lgbm, dev_sub)
    cand = (ph + pl) / 2

    print("\n=== LOCKBOX (154 held-out wells) ===", flush=True)
    print(f"  current  HistGBM full :  {score(truth[lock], blended(cur)):.4f}", flush=True)
    print(f"  cand mean(H,L) step=5 :  {score(truth[lock], blended(cand)):.4f}", flush=True)
    print(f"  (HistGBM step=5 alone :  {score(truth[lock], blended(ph)):.4f})", flush=True)
    print(f"  (LightGBM step=5 alone:  {score(truth[lock], blended(pl)):.4f})", flush=True)


if __name__ == "__main__":
    main()
