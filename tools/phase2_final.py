"""Phase-2 final bake-off: pick the base-learner blend by the LOCKBOX (secondary: pooled CV).
Candidates (all step=5 train subsample): Hist, LGB, Cat singles + their means.

    uv run python -u tools/phase2_final.py
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
import lightgbm as lgb
from catboost import CatBoostRegressor
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
def cat():
    return CatBoostRegressor(iterations=600, learning_rate=0.03, depth=8, l2_leaf_reg=3.0,
                             rsm=0.8, random_seed=0, thread_count=-1, verbose=0)

MODELS = {"Hist": hist, "LGB": lgbm, "Cat": cat}


def main():
    s = load_stack()
    X, truth, anchor, fusion, g = s["X"], s["truth"], s["anchor"], s["fusion"], s["g"]
    y = truth - anchor
    uniq = np.unique(g)
    rng = np.random.default_rng(42)
    lock_wells = set(rng.choice(uniq, int(len(uniq) * 0.2), replace=False).tolist())
    dev = np.array([gi not in lock_wells for gi in g]); lock = ~dev
    sub = train_submask(g, s["first"], s["counts"], 5)
    dev_sub = np.where(dev & sub)[0]
    def bl(p): return 0.6 * (anchor[lock] + p) + 0.4 * fusion[lock]

    # lockbox single-model preds
    preds = {}
    for name, mk in MODELS.items():
        log(f"lockbox fit {name}...")
        m = mk(); m.fit(X[dev_sub], y[dev_sub]); preds[name] = m.predict(X[lock])
    print("\n=== LOCKBOX (154 held-out wells), step=5 ===", flush=True)
    combos = [("Hist",), ("LGB",), ("Cat",), ("Hist", "LGB"), ("LGB", "Cat"),
              ("Hist", "Cat"), ("Hist", "LGB", "Cat")]
    lb = {}
    for c in combos:
        p = np.mean([preds[k] for k in c], axis=0)
        lb[c] = score(truth[lock], bl(p))
        print(f"  {'+'.join(c):16s} {lb[c]:.4f}", flush=True)
    best = min(lb, key=lb.get)
    print(f"  -> best lockbox: {'+'.join(best)} = {lb[best]:.4f}", flush=True)

    # pooled 5-fold CV for the finalists (best + all-three + LGB)
    finalists = {best, ("Hist", "LGB", "Cat"), ("LGB",)}
    log("pooled 5-fold CV for finalists...")
    oofs = {name: np.zeros(len(y)) for name in MODELS}
    for tr, va in GroupKFold(5).split(X, groups=g):
        ti = tr[sub[tr]]
        for name, mk in MODELS.items():
            m = mk(); m.fit(X[ti], y[ti]); oofs[name][va] = m.predict(X[va])
    def blf(p): return 0.6 * (anchor + p) + 0.4 * fusion
    print("\n=== pooled 773-well CV, step=5 (finalists) ===", flush=True)
    for c in finalists:
        p = np.mean([oofs[k] for k in c], axis=0)
        print(f"  {'+'.join(c):16s} {score(truth, blf(p)):.4f}", flush=True)


if __name__ == "__main__":
    main()
