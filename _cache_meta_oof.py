"""Cache the OOF stack predictions used by the sequence-model / postproc / interpretation probes:
/tmp/rogii_meta_oof.npz — truth, anchor, fusion, g, oof_gbm, oof_mlp, hmm, pf, md_since.

This was previously built ad-hoc and was LOST when /tmp was wiped (notes/27). Now scripted so the
local probe rig is reproducible. Requires the base caches first:
    uv run python -u _cache_base.py && uv run python -u _cache_frontier.py \
      && uv run python -u _cache_richfeat.py && uv run python -u _cache_formation.py
    uv run python -u _cache_meta_oof.py            # ~1h (grouped 5-fold OOF of LGB+Cat+MLP)
"""
from __future__ import annotations
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import lightgbm as lgb
from catboost import CatBoostRegressor
from tools.frontier_fast import load_stack, train_submask
import os as _os
_ROGII_CACHE = _os.environ.get("ROGII_CACHE", _os.path.expanduser("~/rogii_cache"))
_os.makedirs(_ROGII_CACHE, exist_ok=True)

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)


def main():
    s = load_stack()
    X0, truth, anchor, fusion, g = s["X"], s["truth"], s["anchor"], s["fusion"], s["g"]
    first, counts = s["first"], s["counts"]
    y = truth - anchor
    r = np.load(f"{_ROGII_CACHE}/rogii_rich.npz")
    Xr = np.nan_to_num(np.column_stack([X0] + [r[k] for k in r.files]), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    fo = np.load(f"{_ROGII_CACHE}/rogii_formation.npz")
    fcols = []
    for k in fo.files:
        v = fo[k].astype(np.float32)
        if k.startswith("tvtF") or k == "tvt_dense":
            v = v - anchor.astype(np.float32)
        fcols.append(v)
    X = np.nan_to_num(np.column_stack([Xr] + fcols), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    sub = train_submask(g, first, counts, 5)
    log(f"X {X.shape}, {len(y)} rows, {len(np.unique(g))} wells")

    oof_gbm = np.zeros(len(y)); oof_mlp = np.zeros(len(y))
    for fold, (tr, va) in enumerate(GroupKFold(5).split(X, groups=g)):
        ti = tr[sub[tr]]
        gp = np.zeros(len(va))
        for sd in (0, 1, 2):
            gp += lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=63,
                    min_child_samples=200, reg_lambda=1.0, subsample=0.8, subsample_freq=1,
                    colsample_bytree=0.8, random_state=sd, n_jobs=-1, verbose=-1).fit(X[ti], y[ti]).predict(X[va])
            gp += CatBoostRegressor(iterations=600, learning_rate=0.03, depth=8, l2_leaf_reg=3.0,
                    rsm=0.8, random_seed=sd, thread_count=-1, verbose=0).fit(X[ti], y[ti]).predict(X[va])
        oof_gbm[va] = gp / 6
        sc = StandardScaler().fit(X[ti])
        tm = np.arange(len(ti)) if len(ti) <= 200000 else np.random.default_rng(0).choice(len(ti), 200000, replace=False)
        mlps = [MLPRegressor(hidden_layer_sizes=(256, 128, 64), activation="relu", alpha=1e-3,
                             batch_size=4096, learning_rate_init=1e-3, early_stopping=True,
                             n_iter_no_change=8, max_iter=120, random_state=sd).fit(sc.transform(X[ti][tm]), y[ti][tm])
                for sd in (0, 1)]
        oof_mlp[va] = np.mean([m.predict(sc.transform(X[va])) for m in mlps], 0)
        log(f"fold {fold} done")

    d = np.load(f"{_ROGII_CACHE}/rogii_feats.npz"); pf = np.load(f"{_ROGII_CACHE}/rogii_pf.npz")
    out = dict(truth=truth, anchor=anchor, fusion=fusion, g=g,
               oof_gbm=oof_gbm.astype(np.float32), oof_mlp=oof_mlp.astype(np.float32),
               hmm=d["hmm_pred"].astype(np.float32), pf=pf["pf_pred"].astype(np.float32),
               md_since=d["md_from_ps"].astype(np.float32))
    np.savez_compressed(f"{_ROGII_CACHE}/rogii_meta_oof.npz", **out)
    from src.validation import score
    pred = 0.75 * (anchor + 0.7 * oof_gbm + 0.3 * oof_mlp) + 0.25 * fusion
    log(f"saved {_ROGII_CACHE}/rogii_meta_oof.npz | baseline pooled OOF = {score(truth, pred):.4f} (expect ~10.90)")


if __name__ == "__main__":
    main()
