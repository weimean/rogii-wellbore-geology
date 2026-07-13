"""The one untested idea from the 7.016 notebook: spatially impute the formation-top SURFACES
from (X,Y) (trained on train-only tops), then reconstruct TVT geometrically. Connects to our
tops probe (TVT = f(Z, tops) per-well R^2=1.0): the missing piece was the eval-zone tops, which
X,Y-imputation can supply (we have eval X,Y). Per-well frame fit on the visible prefix (test-legal).

Pipeline (honest, GroupKFold by well so the imputer never sees the target well):
  1) spatial KNN impute tops(X,Y) for held-out wells (all their rows).
  2) per well, fit TVT ~ [Z, tops] on the KNOWN prefix (Ridge), extrapolate to eval.
Compare: Z-only | ORACLE true tops | IMPUTED tops. Ref: ffill 15.91, our best 13.52.

    uv run python -u tools/spatial_tops.py
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
from src.data import load_split, TARGET_INPUT, FORMATION_DEPTHS

t0 = time.time()
def log(m): print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


def main():
    wells = load_split(Path("data/raw/train"))
    X, Y, Z, TVT, TVTin, gid, km, em = [], [], [], [], [], [], [], []
    tops = []
    gi = 0
    for w in wells:
        h = w.horizontal
        if "TVT" not in h.columns or not all(c in h.columns for c in FORMATION_DEPTHS):
            continue
        m = w.eval_mask.to_numpy(); kn = h[TARGET_INPUT].notna().to_numpy()
        if m.sum() < 10 or kn.sum() < 200:
            continue
        n = len(h)
        X.append(h["X"].to_numpy(float)); Y.append(h["Y"].to_numpy(float))
        Z.append(h["Z"].to_numpy(float)); TVT.append(h["TVT"].to_numpy(float))
        TVTin.append(h[TARGET_INPUT].to_numpy(float))
        tops.append(np.column_stack([h[c].to_numpy(float) for c in FORMATION_DEPTHS]))
        gid.append(np.full(n, gi)); km.append(kn); em.append(m); gi += 1
    X = np.concatenate(X); Y = np.concatenate(Y); Z = np.concatenate(Z)
    TVT = np.concatenate(TVT); TVTin = np.concatenate(TVTin); gid = np.concatenate(gid)
    km = np.concatenate(km); em = np.concatenate(em); TOP = np.vstack(tops)
    XY = np.column_stack([X, Y])
    log(f"{gid.max()+1} wells, {len(X)} rows")

    # --- 1) spatial KNN imputation of tops, OOF by well ---
    imp = np.full_like(TOP, np.nan)
    K = 12
    for tr, va in GroupKFold(5).split(XY, groups=gid):
        # training pool: subsample ~40 rows/well for speed
        pool = []
        for wj in np.unique(gid[tr]):
            idx = tr[gid[tr] == wj]
            pool.append(idx[np.linspace(0, len(idx) - 1, min(40, len(idx))).astype(int)])
        pool = np.concatenate(pool)
        tree = cKDTree(XY[pool])
        dist, nn = tree.query(XY[va], k=K)
        for j in range(TOP.shape[1]):
            imp[va, j] = np.nanmean(TOP[pool][:, j][nn], axis=1)
    top_rmse = np.sqrt(np.nanmean((imp[em] - TOP[em]) ** 2, axis=0))
    log(f"eval-zone top imputation RMSE per formation (ft): {np.round(top_rmse,1)}")

    # --- 2) per-well reconstruct: fit TVT ~ [Z, tops] on KNOWN, extrapolate to eval ---
    def reconstruct(top_src):
        # fill non-finite tops per column with global median (formation absent in some wells)
        ts = top_src.copy()
        for j in range(ts.shape[1]):
            col = ts[:, j]; med = np.nanmedian(col)
            col[~np.isfinite(col)] = med if np.isfinite(med) else 0.0
        pred = np.zeros(len(TVT))
        for wj in range(gid.max() + 1):
            w = gid == wj; kw = w & km & np.isfinite(TVT); ew = w & em
            if kw.sum() < 30:
                lastv = TVTin[w & km][-1] if (w & km).sum() else np.nanmedian(TVT)
                pred[ew] = lastv
                continue
            Fk = np.column_stack([Z[kw], ts[kw]])
            Fe = np.column_stack([Z[ew], ts[ew]])
            m = Ridge(1.0).fit(Fk, TVT[kw])
            pk = m.predict(Fe)
            pred[ew] = np.where(np.isfinite(pk), pk, TVTin[kw][-1])
        return pred

    # their ACTUAL formula: TVT = (top_c - Z) + bias_c, fixed coef 1, bias fit on prefix;
    # per well pick the formation with the best prefix RMSE. Robust to per-well systematic top bias.
    def reconstruct_geom(top_src):
        ts = top_src.copy()
        for j in range(ts.shape[1]):
            col = ts[:, j]; med = np.nanmedian(col); col[~np.isfinite(col)] = med if np.isfinite(med) else 0.0
        pred = np.zeros(len(TVT))
        for wj in range(gid.max() + 1):
            w = gid == wj; kw = w & km & np.isfinite(TVT); ew = w & em
            if kw.sum() < 30:
                pred[ew] = TVTin[w & km][-1] if (w & km).sum() else np.nanmedian(TVT); continue
            best = (np.inf, None)
            for j in range(ts.shape[1]):
                base_k = ts[kw, j] - Z[kw]
                bias = np.mean(TVT[kw] - base_k)
                pr_k = base_k + bias
                e = np.sqrt(np.mean((pr_k - TVT[kw]) ** 2))     # prefix RMSE (visible)
                if e < best[0]:
                    best = (e, (j, bias))
            j, bias = best[1]
            pred[ew] = (ts[ew, j] - Z[ew]) + bias
        return pred

    y = TVT[em]
    for name, src in [("Z-only(ridge)", np.zeros_like(TOP)), ("ORACLE tops(ridge)", TOP),
                      ("IMPUTED tops(ridge)", imp)]:
        p = reconstruct(src)[em]; p = np.where(np.isfinite(p), p, np.nanmedian(y))
        log(f"reconstruct [{name}]: pooled eval RMSE = {np.sqrt(np.mean((y-p)**2)):.3f}")
    for name, src in [("ORACLE tops(GEOM)", TOP), ("IMPUTED tops(GEOM)", imp)]:
        p = reconstruct_geom(src)[em]; p = np.where(np.isfinite(p), p, np.nanmedian(y))
        log(f"reconstruct [{name}]: pooled eval RMSE = {np.sqrt(np.mean((y-p)**2)):.3f}")
    print("  (ref: ffill 15.91, our best gbm_stack 13.52)", flush=True)


if __name__ == "__main__":
    main()
