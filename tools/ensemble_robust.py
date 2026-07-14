"""#2 robustness: does seed-averaging the boosters and/or re-adding HistGBM give a more robust
final than single-seed LGB+Cat? Judged on the lockbox (154 held-out wells), step=5. Cheap
variance reduction for the private LB. dev-fit -> predict lockbox (single fit each).

    uv run python -u tools/ensemble_robust.py
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
import lightgbm as lgb
from catboost import CatBoostRegressor
from tools.frontier_fast import load_stack, train_submask
from src.validation import score

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

def lgbm(seed):
    return lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=63,
                             min_child_samples=200, reg_lambda=1.0, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.8, random_state=seed, n_jobs=-1, verbose=-1)
def cat(seed):
    return CatBoostRegressor(iterations=600, learning_rate=0.03, depth=8, l2_leaf_reg=3.0,
                             rsm=0.8, random_seed=seed, thread_count=-1, verbose=0)
def hist(seed):
    return HistGradientBoostingRegressor(max_iter=600, learning_rate=0.03, max_leaf_nodes=63,
                                         min_samples_leaf=200, l2_regularization=1.0, random_state=seed)


def main():
    s = load_stack()
    X, truth, anchor, fusion, g = s["X"], s["truth"], s["anchor"], s["fusion"], s["g"]
    y = truth - anchor
    sub = train_submask(g, s["first"], s["counts"], 5)
    uniq = np.unique(g); rng = np.random.default_rng(42)
    lock_wells = set(rng.choice(uniq, int(len(uniq) * 0.2), replace=False).tolist())
    dev = np.array([gi not in lock_wells for gi in g]); lock = ~dev
    di = np.where(dev & sub)[0]
    def bl(p): return 0.6 * (anchor[lock] + p) + 0.4 * fusion[lock]

    P = {}  # cache each base learner's lockbox residual-pred
    def fit(tag, mk, seed):
        key = (tag, seed)
        if key not in P:
            log(f"fit {tag} seed{seed}...")
            P[key] = mk(seed).fit(X[di], y[di]).predict(X[lock])
        return P[key]

    seeds = (0, 1, 2)
    lg = [fit("lgb", lgbm, s) for s in seeds]
    ct = [fit("cat", cat, s) for s in seeds]
    hs0 = fit("hist", hist, 0)

    def sc(p): return score(truth[lock], bl(p))
    print("\n=== LOCKBOX (step=5, dev-fit) ===", flush=True)
    print(f"  LGB+Cat  seed0 (current)   : {sc(0.5*(lg[0]+ct[0])):.4f}", flush=True)
    print(f"  LGB+Cat  seed-ens {{0,1,2}}  : {sc((sum(lg)+sum(ct))/6):.4f}", flush=True)
    print(f"  Hist+LGB+Cat seed0         : {sc((hs0+lg[0]+ct[0])/3):.4f}", flush=True)
    print(f"  Hist+LGB+Cat seed-ens      : {sc((hs0+sum(lg)/3+sum(ct)/3)/3):.4f}", flush=True)
    # per-seed spread of the single-config (variance indicator)
    singles = [sc(0.5*(lg[i]+ct[i])) for i in range(3)]
    print(f"  LGB+Cat per-seed spread    : {min(singles):.4f}..{max(singles):.4f} (range {max(singles)-min(singles):.4f})", flush=True)


if __name__ == "__main__":
    main()
