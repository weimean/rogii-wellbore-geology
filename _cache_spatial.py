"""Build spatial-formation features aligned to /tmp/rogii_feats.npz rows.

For each eval row of each kept well (same load_split order + keep logic as _cache_base):
  sp_drift  = boundary-anchored spatial TVT prediction - last_tvt   (predicted eval drift)
  sp_nndist = nearest OTHER-well distance at that row (log1p)        (trust gate)
  sp_impd   = imputed-surface drift from PS boundary                 (raw structural drift)

Imputation: KNN(k=8, 1/(d+1)) on mean formation surface, excluding same-well points (LOWO ->
CV-honest; at inference the cloud is all train wells). Formation surfaces are TRAIN-only but
globally consistent, so imputing from neighbors at this well's X,Y is test-legal.

    uv run python -u _cache_spatial.py
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
from src.data import load_split
from src.model import _stack_features

t0 = time.time()
def log(m): print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)
forms = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]


def main():
    wells = load_split(Path("data/raw/train"))
    # reference cloud: (X,Y) -> mean formation surface, subsampled ::5, tagged by well index k
    pts = []; surf = []; wid = []
    for k, w in enumerate(wells):
        h = w.horizontal
        if not all(f in h.columns for f in forms):
            continue
        X = h["X"].to_numpy(); Y = h["Y"].to_numpy(); F = h[forms].to_numpy()
        m = np.isfinite(X) & np.isfinite(Y) & np.isfinite(F).all(1)
        idx = np.where(m)[0][::5]
        if len(idx) == 0:
            continue
        pts.append(np.column_stack([X[idx], Y[idx]])); surf.append(F[idx].mean(1))
        wid.append(np.full(len(idx), k))
    pts = np.vstack(pts); surf = np.concatenate(surf); wid = np.concatenate(wid)
    tree = cKDTree(pts)
    log(f"cloud {len(pts)} pts; iterating wells")

    out = {"sp_drift": [], "sp_nndist": [], "sp_impd": []}
    gi = 0
    for k, w in enumerate(wells):
        em = w.eval_mask.to_numpy()
        if em.sum() == 0:
            continue
        n = int(em.sum())
        f = _stack_features(w, None, anchor_override=np.zeros(n))
        if f is None or "truth" not in f:
            continue
        h = w.horizontal
        X = h["X"].to_numpy(); Y = h["Y"].to_numpy(); Z = h["Z"].to_numpy()
        Ti = h["TVT_input"].to_numpy()
        last_tvt = f["ff"][em]                 # ffill last-known TVT at eval rows
        drift = np.zeros(n); nnd = np.full(n, 1e6); impd = np.zeros(n)
        has_form = all(ff in h.columns for ff in forms)
        # boundary = last finite TVT_input row
        pp = np.isfinite(Ti)
        if pp.any():
            b = np.where(pp)[0][-1]
            qrows = np.append(np.where(em)[0], b)          # eval rows + boundary
            valid = np.isfinite(X[qrows]) & np.isfinite(Y[qrows]) & np.isfinite(Z[qrows])
            if valid.any():
                q = np.column_stack([X[qrows], Y[qrows]])
                dd, ii = tree.query(np.nan_to_num(q, nan=1e9), k=40)
                same = wid[ii] == k
                dd = np.where(same, np.inf, dd)
                order = np.argsort(dd, axis=1)
                dd = np.take_along_axis(dd, order, 1)[:, :8]
                ii = np.take_along_axis(ii, order, 1)[:, :8]
                wgt = 1.0 / (dd + 1.0); wsum = wgt.sum(1, keepdims=True)
                imp = (wgt * surf[ii]).sum(1) / wsum[:, 0]
                imp = np.where(np.isfinite(imp), imp, np.nan)
                imp_ev = imp[:n]; imp_bnd = imp[n]
                nn_ev = dd[:n, 0]
                Zb = Z[b]
                Db = Ti[b] + Zb                            # test-legal boundary datum
                # boundary-anchored spatial prediction
                sp_pred = imp_ev + (Db - imp_bnd) - Z[np.where(em)[0]]
                ok = np.isfinite(sp_pred)
                drift = np.where(ok, sp_pred - last_tvt, 0.0)
                impd = np.where(np.isfinite(imp_ev - imp_bnd), imp_ev - imp_bnd, 0.0)
                nnd = np.where(np.isfinite(nn_ev), nn_ev, 1e6)
        out["sp_drift"].append(drift.astype(np.float32))
        out["sp_nndist"].append(np.log1p(nnd).astype(np.float32))
        out["sp_impd"].append(impd.astype(np.float32))
        gi += 1
        if (k + 1) % 150 == 0:
            log(f"  {k+1}/{len(wells)} ({gi} kept)")
    out = {kk: np.concatenate(v) for kk, v in out.items()}
    ref = np.load("/tmp/rogii_feats.npz")["truth"]
    assert len(out["sp_drift"]) == len(ref), f"align mismatch {len(out['sp_drift'])} vs {len(ref)}"
    np.savez_compressed("/tmp/rogii_spatial_feat.npz", **out)
    nz = np.mean(out["sp_drift"] != 0)
    log(f"saved {len(out['sp_drift'])} rows; nonzero drift frac {nz:.3f}")


if __name__ == "__main__":
    main()
