"""Residual analysis of the current best (phase-2 LGB+Cat, no geology): where does the
unexplained variance live, and is any of it capturable from available signals?

  1) OOF current-model residual = truth - pred (cached to /tmp/rogii_oof.npz).
  2) Residual RMSE by eval-position bin and eval-length bin (does error grow deeper?).
  3) |residual| correlation with each available per-row feature.
  4) GEOSTEERING test: build NEW features the current model lacks -- steering |d(dz)|, GR-change,
     and their interaction -- and fit an OOF GBM to predict the residual from them. R^2>0 => there
     is un-mined structure (the geosteering hypothesis). R^2~0 => ceiling.
  5) Worst-15 wells and what characterizes them.

    uv run python -u tools/residual_diag.py
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingRegressor
import lightgbm as lgb
from catboost import CatBoostRegressor
from tools.frontier_fast import load_stack, train_submask

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

def lgbm():
    return lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=63,
                             min_child_samples=200, reg_lambda=1.0, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.8, random_state=0, n_jobs=-1, verbose=-1)
def cat():
    return CatBoostRegressor(iterations=600, learning_rate=0.03, depth=8, l2_leaf_reg=3.0,
                             rsm=0.8, random_seed=0, thread_count=-1, verbose=0)

OOF_CACHE = "/tmp/rogii_oof.npz"


def per_well_grad(a, first, counts):
    """gradient of a within each well (rows grouped & MD-ordered)."""
    out = np.zeros_like(a, dtype=float)
    for i in range(len(first)):
        s, e = first[i], first[i] + counts[i]
        out[s:e] = np.gradient(a[s:e]) if e - s > 1 else 0.0
    return out


def main():
    s = load_stack()
    X, truth, anchor, fusion, g = s["X"], s["truth"], s["anchor"], s["fusion"], s["g"]
    first, counts = s["first"], s["counts"]
    y = truth - anchor
    sub = train_submask(g, first, counts, 5)

    if os.path.exists(OOF_CACHE):
        pred = np.load(OOF_CACHE)["pred"]; log("loaded cached OOF pred")
    else:
        log("computing OOF current-model predictions (LGB+Cat step=5)...")
        p = np.zeros(len(y))
        for tr, va in GroupKFold(5).split(X, groups=g):
            ti = tr[sub[tr]]
            pl = lgbm().fit(X[ti], y[ti]).predict(X[va])
            pc = cat().fit(X[ti], y[ti]).predict(X[va])
            p[va] = 0.5 * (pl + pc)
        pred = 0.6 * (anchor + p) + 0.4 * fusion
        np.savez_compressed(OOF_CACHE, pred=pred); log("cached OOF pred")

    resid = truth - pred
    ares = np.abs(resid)
    print(f"\n  overall RMSE = {np.sqrt(np.mean(resid**2)):.3f}   |resid| mean = {ares.mean():.3f}", flush=True)

    d = np.load("/tmp/rogii_feats.npz"); fr = np.load("/tmp/rogii_frontier.npz"); pf = np.load("/tmp/rogii_pf.npz")
    ef = d["eval_frac"]

    # (2) residual RMSE by eval-position
    print("\n=== residual RMSE by eval position (0=just after PS, 1=well toe) ===", flush=True)
    for lo in np.arange(0, 1, 0.2):
        m = (ef >= lo) & (ef < lo + 0.2)
        print(f"  frac [{lo:.1f},{lo+0.2:.1f}): RMSE={np.sqrt(np.mean(resid[m]**2)):.3f}  (n={m.sum()})", flush=True)

    # (3) |resid| correlation with available per-row features
    feats = {
        "eval_frac": ef, "z_from_ps": d["z_from_ps"], "md_from_ps": d["md_from_ps"],
        "horiz_from_ps": d["horiz_from_ps"], "dz(incl)": d["dz"], "tort": d["tort"],
        "gr50": d["gr50"], "gr_dev": d["gr_dev"], "hmm_std": d["hmm_std"], "pf_std": pf["pf_std"],
        "hmm-pf disagree": np.abs(d["hmm_pred"] - pf["pf_pred"]), "gr_env_std31": fr["gstd31"],
        "gabsd(|dGR|)": fr["gabsd"],
    }
    print("\n=== corr( feature , |residual| )  [what predicts error magnitude] ===", flush=True)
    for k, v in sorted(feats.items(), key=lambda kv: -abs(np.corrcoef(kv[1], ares)[0, 1])):
        print(f"  {k:18s} {np.corrcoef(v, ares)[0,1]:+.3f}", flush=True)

    # (4) GEOSTEERING interaction: new features the current model lacks
    log("building geosteering interaction features...")
    incl = d["dz"]                                  # dZ/dMD (inclination)
    steer = np.abs(per_well_grad(incl, first, counts))   # |d(incl)| = dogleg/steering
    grchg = fr["gabsd"]                             # |dGR| (smoothed)
    def z(a): return (a - a.mean()) / (a.std() + 1e-9)
    inter = z(steer) * z(grchg)                     # steering x GR-change
    grchg_grad = np.abs(per_well_grad(grchg, first, counts))
    NEW = np.column_stack([steer, grchg, inter, grchg_grad, z(steer) * z(np.abs(d["gr_dev"]))])

    print("\n=== corr( geosteering feature , residual & |residual| ) ===", flush=True)
    for name, col in [("steer|d(incl)|", steer), ("GRchange", grchg), ("steer x GRchg", inter)]:
        print(f"  {name:16s} vs resid {np.corrcoef(col, resid)[0,1]:+.3f}   vs |resid| {np.corrcoef(col, ares)[0,1]:+.3f}", flush=True)

    # decisive: can an OOF GBM predict the residual from these NEW features?
    log("OOF GBM: predict residual from NEW geosteering features...")
    rp = np.zeros(len(resid))
    for tr, va in GroupKFold(5).split(NEW, groups=g):
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                          min_samples_leaf=500, random_state=0)
        m.fit(NEW[tr[sub[tr]]], resid[tr[sub[tr]]]); rp[va] = m.predict(NEW[va])
    r2 = 1 - np.sum((resid - rp)**2) / np.sum((resid - resid.mean())**2)
    print(f"\n  >>> residual-R^2 from NEW geosteering feats (OOF) = {r2:+.4f}", flush=True)
    print(f"      (>0 => un-mined structure; ~0 => at ceiling)", flush=True)

    # (5) worst wells
    uniq = np.unique(g)
    wrmse = np.array([np.sqrt(np.mean(resid[first[i]:first[i]+counts[i]]**2)) for i in range(len(uniq))])
    order = np.argsort(wrmse)[::-1][:15]
    print("\n=== worst 15 wells (by RMSE) — eval_len, known_range, v50 ===", flush=True)
    for i in order:
        s0 = first[i]
        print(f"  well#{uniq[i]:>4d} RMSE={wrmse[i]:6.1f}  eval_len={d['eval_len'][s0]:.0f}  "
              f"known_range={d['known_range'][s0]:.0f}  v50={d['v50'][s0]:+.3f}  gr_off={d['gr_off'][s0]:+.1f}", flush=True)
    print(f"\n  worst-15 mean RMSE {wrmse[order].mean():.1f} vs median well RMSE {np.median(wrmse):.1f}", flush=True)


if __name__ == "__main__":
    main()
