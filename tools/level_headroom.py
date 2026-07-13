"""How much per-well LEVEL (eval-mean TVT) does our FULL stack already recover, and how
much headroom is left before the oracle-constant (9.04)?

The 10% figure is the ANCHOR alone; HMM+PF+GBM add level info on top. This aggregates each
component's OOF prediction to a per-well eval-MEAN and reports R^2 on the true eval-mean and
on the offset (eval-mean - last_known). Uses the cached posteriors (fast, no recompute).

    uv run python -u tools/level_headroom.py
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from sklearn.model_selection import GroupKFold, KFold
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor

t0 = time.time()
def log(m): print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)

STACK_FEATS = ["hmm_pred", "hmm_std", "z_from_ps", "md_from_ps", "horiz_from_ps", "dz",
               "tort", "gr50", "gr_dev", "az_sin", "az_cos", "eval_frac",
               "v50", "v200", "gr_off", "known_range", "eval_len"]
LVL = ["v50", "v200", "gr_off", "eval_len", "known_range"]


def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def per_well_mean(g, uniq, arr):
    return np.array([arr[g == u].mean() for u in uniq])


def main():
    d = np.load("/tmp/rogii_feats.npz"); pf = np.load("/tmp/rogii_pf.npz")
    fr = np.load("/tmp/rogii_frontier.npz")
    g = d["gidx"]; truth = d["truth"]; last = d["last_tvt"]
    hp, hs = d["hmm_pred"], d["hmm_std"]; pp, ps = pf["pf_pred"], pf["pf_std"]
    uniq = np.unique(g)
    log(f"{len(uniq)} wells, {len(truth)} rows")

    # OOF well-level anchor (Ridge on offset, GroupKFold consistent with rows)
    well_X = np.array([[d[k][g == u][0] for k in LVL] for u in uniq])
    well_off = np.array([truth[g == u].mean() - last[g == u][0] for u in uniq])
    well_last = np.array([last[g == u][0] for u in uniq])
    anchor_off = np.zeros(len(uniq))
    for tr, va in KFold(5, shuffle=True, random_state=0).split(well_X):
        m = Ridge(1.0).fit(well_X[tr], well_off[tr]); anchor_off[va] = m.predict(well_X[va])
    off_of = {u: anchor_off[i] for i, u in enumerate(uniq)}
    anchor = last + np.array([off_of[u] for u in g])

    # fusion + full stack (base + A + B), OOF GBM over rows
    wt = 1 / (1 + (hs / 4) ** 2); fusion = wt * hp + (1 - wt) * anchor
    wtp = 1 / (1 + (ps / 4) ** 2); pf_fus = wtp * pp + (1 - wtp) * anchor
    A = np.column_stack([np.abs(hp - pp), hs + ps, np.abs(hp - pp) / (hs + ps + 1),
                         np.abs(hp - anchor), np.abs(pp - anchor)])
    B = np.column_stack([fr[k] for k in ["gmax51", "gmin51", "grms51", "gstd31", "gstd151", "gabsd"]])
    X = np.column_stack([d[k] for k in STACK_FEATS] +
                        [anchor, hp - anchor, fusion, wt, pp, ps, pp - anchor, hp - pp, pf_fus])
    X = np.column_stack([X, A, B])
    y = truth - anchor
    oof = np.zeros(len(y))
    for tr, va in GroupKFold(5).split(X, groups=g):
        m = HistGradientBoostingRegressor(max_iter=600, learning_rate=0.03, max_leaf_nodes=63,
                                          min_samples_leaf=200, l2_regularization=1.0, random_state=0)
        m.fit(X[tr], y[tr]); oof[va] = m.predict(X[va])
    stack = 0.6 * (anchor + oof) + 0.4 * fusion
    log("OOF stack done")

    # per-well eval-mean recovery
    tv_mean = per_well_mean(g, uniq, truth)
    comps = {
        "ffill (last)":      well_last,
        "anchor (Ridge)":    per_well_mean(g, uniq, anchor),
        "hmm eval-mean":     per_well_mean(g, uniq, hp),
        "fusion eval-mean":  per_well_mean(g, uniq, fusion),
        "FULL stack":        per_well_mean(g, uniq, stack),
    }
    print("\n=== per-well LEVEL recovery (eval-mean TVT) ===", flush=True)
    print(f"  offset std (target spread) = {well_off.std():.2f} ft", flush=True)
    for name, pm in comps.items():
        pred_off = pm - well_last
        print(f"  {name:20s} eval-mean R2={r2(tv_mean, pm):+.3f}   "
              f"offset R2={r2(well_off, pred_off):+.3f}   "
              f"level-RMSE={np.sqrt(np.mean((tv_mean-pm)**2)):.2f}", flush=True)
    print(f"\n  oracle-constant level-RMSE = 0 by definition; competition oracle-const RMSE=9.04",
          flush=True)


if __name__ == "__main__":
    main()
