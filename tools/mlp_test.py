"""Have we tried a neural net? No. Test a plain sklearn MLPRegressor (no torch, runs locally) on
our 91-feature matrix (base + rich). Does the neural function class help alone, and/or add
diversity blended with LGB+Cat? Lockbox, w=0.75. (Not TabICL, but a genuine NN.)

    uv run python -u tools/mlp_test.py
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
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
    uniq = np.unique(g); rng = np.random.default_rng(42)
    lock_wells = set(rng.choice(uniq, int(len(uniq) * 0.2), replace=False).tolist())
    dev = np.array([gi not in lock_wells for gi in g]); lock = ~dev
    di = np.where(dev & sub)[0]
    # subsample dev for MLP speed
    di_m = np.random.default_rng(0).choice(di, min(200000, len(di)), replace=False)
    li = np.where(lock)[0]
    def bl(p): return score(truth[li], 0.75 * (anchor[li] + p) + 0.25 * fusion[li])

    log(f"features {X.shape[1]}, MLP train {len(di_m)}, lock {len(li)}")
    # LGB+Cat baseline (seed-ens) on the SAME dev-sub
    gpred = np.zeros(len(li))
    for sd in (0, 1, 2):
        gpred += lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=63, min_child_samples=200,
                    reg_lambda=1.0, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=sd,
                    n_jobs=-1, verbose=-1).fit(X[di], y[di]).predict(X[li])
        gpred += CatBoostRegressor(iterations=600, learning_rate=0.03, depth=8, l2_leaf_reg=3.0, rsm=0.8,
                    random_seed=sd, thread_count=-1, verbose=0).fit(X[di], y[di]).predict(X[li])
    gpred /= 6
    log("LGB+Cat done")

    # MLP (2 seeds averaged for stability), scaled
    mpred = np.zeros(len(li))
    for sd in (0, 1):
        pipe = make_pipeline(StandardScaler(),
                             MLPRegressor(hidden_layer_sizes=(256, 128, 64), activation="relu",
                                          alpha=1e-3, batch_size=4096, learning_rate_init=1e-3,
                                          early_stopping=True, n_iter_no_change=8, max_iter=120,
                                          random_state=sd))
        pipe.fit(X[di_m], y[di_m]); mpred += pipe.predict(X[li])
        log(f"  MLP seed{sd} done")
    mpred /= 2

    print("\n=== LOCKBOX (w=0.75) ===", flush=True)
    print(f"  LGB+Cat:            {bl(gpred):.4f}", flush=True)
    print(f"  MLP alone:          {bl(mpred):.4f}", flush=True)
    for wm in (0.1, 0.2, 0.3, 0.4, 0.5):
        print(f"  {(1-wm):.1f}*GBM + {wm:.1f}*MLP:  {bl((1-wm)*gpred + wm*mpred):.4f}", flush=True)
    print(f"  corr(GBM,MLP)={np.corrcoef(gpred,mpred)[0,1]:.3f} (lower=more diverse)", flush=True)


if __name__ == "__main__":
    main()
