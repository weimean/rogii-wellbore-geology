"""Meta-stack-of-stacks (notes/15, option 3): replace our fixed blend (0.7 GBM + 0.3 MLP, then
0.75/0.25 fusion) with a positive-Ridge META learned over the component OOF predictions
[p_gbm, p_mlp, fusion, hmm, pf] (all TVT-scale), fit out-of-fold. + optional MD-ramp fade of the
correction near the PS boundary. Compare pooled RMSE vs our current fixed blend. Caches: base+rich+
formation. Saves OOF to /tmp so meta weights can be swept cheaply.

    uv run python -u tools/meta_stack_test.py
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
import lightgbm as lgb
from catboost import CatBoostRegressor
from tools.frontier_fast import load_stack, train_submask
from src.validation import score

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)
OOF = "/tmp/rogii_meta_oof.npz"


def build_oof():
    s = load_stack()
    X0, truth, anchor, fusion, g = s["X"], s["truth"], s["anchor"], s["fusion"], s["g"]
    y = truth - anchor
    r = np.load("/tmp/rogii_rich.npz")
    fo = np.load("/tmp/rogii_formation.npz")
    fcols = []
    for k in fo.files:
        v = fo[k].astype(np.float32)
        if k.startswith("tvtF") or k == "tvt_dense":
            v = v - anchor.astype(np.float32)
        fcols.append(v)
    X = np.column_stack([X0] + [r[k] for k in r.files] + fcols).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    sub = train_submask(g, s["first"], s["counts"], 5)
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPRegressor
    oof_gbm = np.zeros(len(y)); oof_mlp = np.zeros(len(y))
    for fi, (tr, va) in enumerate(GroupKFold(5).split(X, groups=g)):
        ti = tr[sub[tr]]
        for sd in (0, 1, 2):
            oof_gbm[va] += lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=63,
                min_child_samples=200, reg_lambda=1.0, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                random_state=sd, n_jobs=-1, verbose=-1).fit(X[ti], y[ti]).predict(X[va])
            oof_gbm[va] += CatBoostRegressor(iterations=600, learning_rate=0.03, depth=8, l2_leaf_reg=3.0,
                rsm=0.8, random_seed=sd, thread_count=-1, verbose=0).fit(X[ti], y[ti]).predict(X[va])
        Xn = np.nan_to_num(X); scaler = StandardScaler().fit(Xn[ti])
        tm = ti if len(ti) <= 200000 else np.random.default_rng(0).choice(ti, 200000, replace=False)
        oof_mlp[va] = MLPRegressor(hidden_layer_sizes=(256, 128, 64), activation="relu", alpha=1e-3,
            batch_size=4096, learning_rate_init=1e-3, early_stopping=True, n_iter_no_change=8, max_iter=120,
            random_state=0).fit(scaler.transform(Xn[tm]), y[tm]).predict(scaler.transform(Xn[va]))
        log(f"fold {fi} done")
    oof_gbm /= 6
    md_since = X0[:, 3]  # md_from_ps is STACK_FEATS index 3
    np.savez(OOF, oof_gbm=oof_gbm, oof_mlp=oof_mlp, truth=truth, anchor=anchor, fusion=fusion,
             g=g, hmm=X0[:, 0], pf=s["X"][:, 0] * 0 + 0, md_since=md_since)
    return OOF


def main():
    if not os.path.exists(OOF):
        log("building OOF (LGB+Cat+MLP, ~slow)...")
        build_oof()
    d = np.load(OOF)
    truth, anchor, fusion, g = d["truth"], d["anchor"], d["fusion"], d["g"]
    oof_gbm, oof_mlp, md_since = d["oof_gbm"], d["oof_mlp"], d["md_since"]
    p_gbm = anchor + oof_gbm; p_mlp = anchor + oof_mlp
    # current fixed blend
    cur = 0.75 * (anchor + 0.7 * oof_gbm + 0.3 * oof_mlp) + 0.25 * fusion
    print(f"\n=== pooled 773-well CV ===", flush=True)
    print(f"  current fixed blend:   {score(truth, cur):.4f}", flush=True)
    # positive-Ridge META over component TVT-scale predictions, fit OUT-OF-FOLD (nested)
    cols = np.column_stack([p_gbm, p_mlp, fusion])
    meta = np.zeros(len(truth))
    for tr, va in GroupKFold(5).split(cols, groups=g):
        rg = Ridge(alpha=1.0, positive=True).fit(cols[tr], truth[tr])
        meta[va] = rg.predict(cols[va])
    print(f"  positive-Ridge meta:   {score(truth, meta):.4f}   weights≈{np.round(Ridge(alpha=1.0,positive=True).fit(cols,truth).coef_,3)}", flush=True)
    # meta + MD-ramp fade of the correction near PS boundary (tau in MD units)
    for tau in (60.0, 120.0, 240.0):
        ramp = 1.0 - np.exp(-np.maximum(md_since, 0.0) / tau)
        faded = anchor + ramp * (meta - anchor)
        print(f"  meta + MD-ramp fade τ={tau:.0f}: {score(truth, faded):.4f}", flush=True)


if __name__ == "__main__":
    main()
