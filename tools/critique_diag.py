"""Data-science diagnostics for the geology win (turn the critique into evidence):
  (A) Bootstrap 95% CI on the pooled-RMSE delta (resample WELLS) — is -0.53 real vs noise?
  (B) Ablate geo_last_med (absolute TVT): does the gain survive on RELATIVE features only?
      If not, the model is memorizing absolute depth -> won't transfer to a shifted test set.
  (C) Test-well representativeness: are the 3 test wells' geo features in the train distribution?

    uv run python -u tools/critique_diag.py
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
import lightgbm as lgb
from catboost import CatBoostRegressor
from tools.frontier_fast import load_stack, train_submask
from src.data import load_split, TARGET_INPUT
from src.model import hmm_posterior, _geology_features

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

def lgbm():
    return lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=63,
                             min_child_samples=200, reg_lambda=1.0, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.8, random_state=0, n_jobs=-1, verbose=-1)
def cat():
    return CatBoostRegressor(iterations=600, learning_rate=0.03, depth=8, l2_leaf_reg=3.0,
                             rsm=0.8, random_seed=0, thread_count=-1, verbose=0)


def oof(X, y, anchor, fusion, g, sub):
    p = np.zeros(len(y))
    for tr, va in GroupKFold(5).split(X, groups=g):
        ti = tr[sub[tr]]
        pl = lgbm().fit(X[ti], y[ti]).predict(X[va])
        pc = cat().fit(X[ti], y[ti]).predict(X[va])
        p[va] = 0.5 * (pl + pc)
    return 0.6 * (anchor + p) + 0.4 * fusion   # full per-row prediction


def per_well_se(g, uniq, first, counts, truth, pred):
    # returns (sum of sq err, count) per well for pooled-RMSE bootstrap
    sse = np.array([np.sum((truth[first[i]:first[i]+counts[i]] - pred[first[i]:first[i]+counts[i]])**2)
                    for i in range(len(uniq))])
    return sse, counts.astype(float)


def main():
    s = load_stack()
    X0, truth, anchor, fusion, g = s["X"], s["truth"], s["anchor"], s["fusion"], s["g"]
    first, counts = s["first"], s["counts"]; uniq = np.unique(g)
    y = truth - anchor
    sub = train_submask(g, first, counts, 5)
    geo = np.load("/tmp/rogii_geology.npz")
    g_off, g_hmm, g_med = geo["geo_last_off"], geo["geo_hmm_off"], geo["geo_last_med"]
    Xrel = np.column_stack([X0, g_off, g_hmm]).astype(np.float32)          # no absolute median
    Xfull = np.column_stack([X0, g_off, g_hmm, g_med]).astype(np.float32)

    log("OOF: no-geo ..."); p_no = oof(X0, y, anchor, fusion, g, sub)
    log("OOF: rel-geo (no geo_last_med) ..."); p_rel = oof(Xrel, y, anchor, fusion, g, sub)
    log("OOF: full-geo ..."); p_full = oof(Xfull, y, anchor, fusion, g, sub)

    def rmse(pred): return float(np.sqrt(np.mean((truth - pred) ** 2)))
    print("\n=== pooled CV ===", flush=True)
    print(f"  no-geo:   {rmse(p_no):.4f}", flush=True)
    print(f"  rel-geo:  {rmse(p_rel):.4f}   (drop geo_last_med)", flush=True)
    print(f"  full-geo: {rmse(p_full):.4f}", flush=True)

    # (A/B) bootstrap CI on deltas, resampling WELLS
    sse_no, c = per_well_se(g, uniq, first, counts, truth, p_no)
    sse_rel, _ = per_well_se(g, uniq, first, counts, truth, p_rel)
    sse_full, _ = per_well_se(g, uniq, first, counts, truth, p_full)
    nw = len(uniq); rng = np.random.default_rng(0)
    d_full, d_rel = [], []
    for _ in range(2000):
        idx = rng.integers(0, nw, nw)
        tot = c[idx].sum()
        base = np.sqrt(sse_no[idx].sum() / tot)
        d_full.append(np.sqrt(sse_full[idx].sum() / tot) - base)
        d_rel.append(np.sqrt(sse_rel[idx].sum() / tot) - base)
    d_full = np.array(d_full); d_rel = np.array(d_rel)
    print("\n=== bootstrap 95% CI on pooled-RMSE delta vs no-geo (2000 resamples of wells) ===", flush=True)
    print(f"  full-geo delta: {d_full.mean():+.3f}  95% CI [{np.percentile(d_full,2.5):+.3f}, {np.percentile(d_full,97.5):+.3f}]", flush=True)
    print(f"  rel-geo  delta: {d_rel.mean():+.3f}  95% CI [{np.percentile(d_rel,2.5):+.3f}, {np.percentile(d_rel,97.5):+.3f}]", flush=True)
    print(f"  -> geo_last_med (absolute) marginal contribution: {rmse(p_rel)-rmse(p_full):+.3f}", flush=True)

    # (C) test-well representativeness: geo features of the 3 test wells vs train
    log("test-well geo features vs train...")
    train_off = g_off[first]; train_med = g_med[first]   # per-well
    test = load_split(Path("data/raw/test"))
    print("\n=== test-well geology vs train distribution ===", flush=True)
    print(f"  train geo_last_med: [{train_med.min():.0f}, {train_med.max():.0f}]  (p5 {np.percentile(train_med,5):.0f}, p95 {np.percentile(train_med,95):.0f})", flush=True)
    print(f"  train geo_last_off: [{train_off.min():.1f}, {train_off.max():.1f}]  (p5 {np.percentile(train_off,5):.1f}, p95 {np.percentile(train_off,95):.1f})", flush=True)
    for w in test:
        h = w.horizontal; kn = h[h[TARGET_INPUT].notna()]
        if len(kn) < 5:
            continue
        last = float(kn[TARGET_INPUT].values[-1])
        n_em = int(w.eval_mask.to_numpy().sum())
        go, gh, gm = _geology_features(w, np.full(min(n_em, 1), last), last)
        # geo_last_off/med don't need hmm; use last for a single-value probe
        pct_med = float((train_med < gm[0]).mean() * 100)
        pct_off = float((train_off < go[0]).mean() * 100)
        print(f"  {w.name}: last={last:.0f}  geo_last_med={gm[0]:.0f} (train pctile {pct_med:.0f})  "
              f"geo_last_off={go[0]:+.1f} (train pctile {pct_off:.0f})", flush=True)


if __name__ == "__main__":
    main()
