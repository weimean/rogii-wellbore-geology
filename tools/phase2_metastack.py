"""Phase 2 — multi-learner meta-stack (wiggle diversity via different ALGORITHMS).

Current stack corrects the anchor with ONE HistGBM, then blends 0.6/0.4 with fusion. Here we
train diverse base learners on the same feature matrix (target = truth - anchor), plus fusion
as a base, and let a Ridge meta-learner weight them. Model-level diversity is the lever the
'winning principle' predicts *might* help; memory says LightGBM ties HistGBM, so expectation is
marginal. Kept only if it beats the pooled CV (13.540) and the lockbox.

Base learners: HistGBM (current), LightGBM (leaf-wise), ExtraTrees (bagging).
Reports pooled 773-well GroupKFold CV for each single learner and the meta.

    uv run python -u tools/phase2_metastack.py
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from sklearn.model_selection import GroupKFold, KFold
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
import lightgbm as lgb
from src.validation import score

t0 = time.time()
def log(m): print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)

STACK_FEATS = ["hmm_pred", "hmm_std", "z_from_ps", "md_from_ps", "horiz_from_ps", "dz",
               "tort", "gr50", "gr_dev", "az_sin", "az_cos", "eval_frac",
               "v50", "v200", "gr_off", "known_range", "eval_len"]
LVL = ["v50", "v200", "gr_off", "eval_len", "known_range"]


def build():
    d = np.load("/tmp/rogii_feats.npz"); pf = np.load("/tmp/rogii_pf.npz")
    fr = np.load("/tmp/rogii_frontier.npz")
    g = d["gidx"]; truth = d["truth"]; last = d["last_tvt"]
    hp, hs = d["hmm_pred"], d["hmm_std"]; pp, ps = pf["pf_pred"], pf["pf_std"]
    uniq = np.unique(g)
    well_X = np.array([[d[k][g == u][0] for k in LVL] for u in uniq])
    well_off = np.array([truth[g == u].mean() - last[g == u][0] for u in uniq])
    anchor_off = np.zeros(len(uniq))
    for tr, va in KFold(5, shuffle=True, random_state=0).split(well_X):
        anchor_off[va] = Ridge(1.0).fit(well_X[tr], well_off[tr]).predict(well_X[va])
    off_of = {u: anchor_off[i] for i, u in enumerate(uniq)}
    anchor = last + np.array([off_of[u] for u in g])
    wt = 1 / (1 + (hs / 4) ** 2); fusion = wt * hp + (1 - wt) * anchor
    wtp = 1 / (1 + (ps / 4) ** 2); pf_fus = wtp * pp + (1 - wtp) * anchor
    A = np.column_stack([np.abs(hp - pp), hs + ps, np.abs(hp - pp) / (hs + ps + 1),
                         np.abs(hp - anchor), np.abs(pp - anchor)])
    B = np.column_stack([fr[k] for k in ["gmax51", "gmin51", "grms51", "gstd31", "gstd151", "gabsd"]])
    X = np.column_stack([d[k] for k in STACK_FEATS] +
                        [anchor, hp - anchor, fusion, wt, pp, ps, pp - anchor, hp - pp, pf_fus])
    X = np.column_stack([X, A, B])
    return X, truth, anchor, fusion, g


def oof_hist(X, y, g):
    p = np.zeros(len(y))
    for tr, va in GroupKFold(5).split(X, groups=g):
        m = HistGradientBoostingRegressor(max_iter=600, learning_rate=0.03, max_leaf_nodes=63,
                                          min_samples_leaf=200, l2_regularization=1.0, random_state=0)
        m.fit(X[tr], y[tr]); p[va] = m.predict(X[va])
    return p


def oof_lgb(X, y, g):
    p = np.zeros(len(y))
    for tr, va in GroupKFold(5).split(X, groups=g):
        m = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=63,
                              min_child_samples=200, reg_lambda=1.0, subsample=0.8,
                              subsample_freq=1, colsample_bytree=0.8, random_state=0, n_jobs=-1, verbose=-1)
        m.fit(X[tr], y[tr]); p[va] = m.predict(X[va])
    return p


def oof_et(X, y, g):
    p = np.zeros(len(y))
    for tr, va in GroupKFold(5).split(X, groups=g):
        m = ExtraTreesRegressor(n_estimators=200, min_samples_leaf=200, max_features=0.6,
                                n_jobs=-1, random_state=0)
        m.fit(X[tr], y[tr]); p[va] = m.predict(X[va])
    return p


def main():
    log("building features...")
    X, truth, anchor, fusion, g = build()
    y = truth - anchor
    log(f"{len(y)} rows; training base learners (OOF GroupKFold)...")

    hist = oof_hist(X, y, g); log("  hist done")
    lg = oof_lgb(X, y, g); log("  lgb done")

    # single-learner stacks (each blended 0.6/0.4 with fusion, matching production)
    def blended(gbm): return 0.6 * (anchor + gbm) + 0.4 * fusion
    print("\n=== pooled 773-well GroupKFold CV ===", flush=True)
    print(f"  HistGBM (current):    {score(truth, blended(hist)):.3f}", flush=True)
    print(f"  LightGBM:             {score(truth, blended(lg)):.3f}", flush=True)
    print(f"  mean(Hist,LGB):       {score(truth, blended((hist+lg)/2)):.3f}", flush=True)

    # Ridge meta over base preds + fusion (residual space), OOF via 2nd GroupKFold
    fus_res = fusion - anchor
    base = np.column_stack([hist, lg, fus_res])
    meta = np.zeros(len(y))
    for tr, va in GroupKFold(5).split(base, groups=g):
        m = Ridge(1.0, positive=True).fit(base[tr], y[tr]); meta[va] = m.predict(base[va])
    mfull = Ridge(1.0, positive=True).fit(base, y)
    print(f"  Ridge-meta (H,L,fus): {score(truth, anchor + meta):.3f}  "
          f"weights={np.round(mfull.coef_,3)}", flush=True)


if __name__ == "__main__":
    main()
