"""Phase-1 LEVEL probe: how much of the per-well offset variance can we recover, and
does spatial K-NN (family 2) carry real signal or is it a leak-trap?

Offset target per well = mean(eval-zone TVT) - last_known_TVT  (std ~12.7 ft, 68% of error).
Current anchor (5-feat Ridge) recovers ~10% of its variance. We want to beat that.

Reports OOF R^2 (variance explained) over 773 wells, 5-fold GroupKFold, for:
  - baseline : current 5 level features (v50,v200,gr_off,eval_len,known_range), Ridge
  - spatial  : predict offset = mean offset of K nearest OTHER wells by PS (X,Y)
  - combined : baseline features + spatial-neighbor offset, Ridge
Also prints the spatial interspersion stats (NN distances) that gate family 2.

    uv run python -u tools/level_probe.py
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from src.data import load_split, TARGET_INPUT
from src.model import LEVEL_FEATURES, extract_level_features, _gr_offset_raw

t0 = time.time()
def log(m): print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


def build_table():
    wells = load_split(Path("data/raw/train"))
    rows = []
    for w in wells:
        h = w.horizontal
        em = w.eval_mask.to_numpy()
        kn = h[h[TARGET_INPUT].notna()]
        if em.sum() == 0 or len(kn) < 200 or "TVT" not in h.columns:
            continue
        kt = kn[TARGET_INPUT].values
        last = float(kt[-1])
        ev_tvt = h["TVT"].to_numpy(float)[em]
        offset = float(np.nanmean(ev_tvt) - last)
        # PS point (last known) X,Y — spatial anchor for the lateral
        psx = float(kn["X"].iloc[-1]); psy = float(kn["Y"].iloc[-1])
        feats = extract_level_features(w)  # dict of the 5 LEVEL_FEATURES
        rows.append(dict(name=w.name, offset=offset, psx=psx, psy=psy,
                         **{k: float(feats[k]) for k in LEVEL_FEATURES}))
    return rows


def r2(y, p):
    ss_res = np.sum((y - p) ** 2); ss_tot = np.sum((y - y.mean()) ** 2)
    return 1 - ss_res / ss_tot


def main():
    log("building well-level table...")
    rows = build_table()
    n = len(rows)
    y = np.array([r["offset"] for r in rows])
    XY = np.array([[r["psx"], r["psy"]] for r in rows])
    Xb = np.array([[r[k] for k in LEVEL_FEATURES] for r in rows])
    log(f"{n} wells; offset mean={y.mean():.2f} std={y.std():.2f}")

    # --- spatial interspersion: nearest-neighbour distances between well PS points ---
    from scipy.spatial import cKDTree
    tree = cKDTree(XY)
    dists, idx = tree.query(XY, k=2)  # [:,0]=self(0), [:,1]=nearest other
    nn = dists[:, 1]
    log(f"NN dist (ft): median={np.median(nn):.0f}  p10={np.percentile(nn,10):.0f}  "
        f"p90={np.percentile(nn,90):.0f}  max={nn.max():.0f}")

    kf = KFold(5, shuffle=True, random_state=0)
    base_oof = np.zeros(n); spat_oof = np.zeros(n); comb_oof = np.zeros(n)
    K = 5
    for tr, va in kf.split(Xb):
        # baseline Ridge
        m = Ridge(1.0).fit(Xb[tr], y[tr]); base_oof[va] = m.predict(Xb[va])
        # spatial: mean offset of K nearest TRAIN wells (no leakage — neighbors are train only)
        ttree = cKDTree(XY[tr])
        d, ii = ttree.query(XY[va], k=min(K, len(tr)))
        spat_oof[va] = y[tr][ii].mean(axis=1)
        # combined
        Xc_tr = np.column_stack([Xb[tr], _neighbor_pred(XY, y, tr, tr, K)])
        Xc_va = np.column_stack([Xb[va], spat_oof[va]])
        mc = Ridge(1.0).fit(Xc_tr, y[tr]); comb_oof[va] = mc.predict(Xc_va)

    print("\n=== per-well offset OOF R^2 (variance explained), 5-fold ===", flush=True)
    print(f"  baseline (5 feats):     R2={r2(y, base_oof):+.3f}", flush=True)
    print(f"  spatial K-NN (K={K}):     R2={r2(y, spat_oof):+.3f}", flush=True)
    print(f"  combined:               R2={r2(y, comb_oof):+.3f}", flush=True)
    print(f"  (corr spatial vs truth: {np.corrcoef(spat_oof, y)[0,1]:+.3f})", flush=True)


def _neighbor_pred(XY, y, fit_idx, query_idx, K):
    """Leave-one-out neighbour offset within a training fold (for the combined model's
    training matrix): each fit well's spatial feature = mean of its K nearest OTHER fit wells."""
    from scipy.spatial import cKDTree
    t = cKDTree(XY[fit_idx])
    d, ii = t.query(XY[query_idx], k=min(K + 1, len(fit_idx)))
    return y[fit_idx][ii[:, 1:]].mean(axis=1)  # drop self (col 0)


if __name__ == "__main__":
    main()
