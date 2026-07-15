"""Spatial formation-surface imputation: can neighbor wells' structural surfaces (ANCC..BUDA,
TRAIN-only but globally consistent) predict the eval-zone datum drift when imputed at a test
well's X,Y? Test-legal (neighbor surfaces + own X,Y; never own formation cols). LOWO on train.

Finding so far: oracle surfaces -> RMSE 0.01 (TVT is deterministic in the surfaces). KNN imputation
error ~18 ft loses to ffill for far wells, but wells with a CLOSE neighbor (<50 ft) go 12->3.7.
This sizes the pooled, distance-gated gain over all 765 wells.

    uv run python -u tools/spatial_formation_probe.py
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
from src.data import load_split

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

forms = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]


def main():
    wells = load_split(Path("data/raw/train"))
    pts = []; surf = []; wid = []; data = []
    for wi, w in enumerate(wells):
        h = w.horizontal
        if not all(f in h.columns for f in forms):
            continue
        X = h["X"].to_numpy(); Y = h["Y"].to_numpy(); Z = h["Z"].to_numpy()
        T = h["TVT"].to_numpy(); Ti = h["TVT_input"].to_numpy(); F = h[forms].to_numpy()
        m = np.isfinite(X) & np.isfinite(Y) & np.isfinite(Z) & np.isfinite(T) & np.isfinite(F).all(1)
        if m.sum() < 80:
            continue
        data.append((wi, X, Y, Z, T, Ti, F, m))
        idx = np.where(m)[0][::5]
        pts.append(np.column_stack([X[idx], Y[idx]])); surf.append(F[idx].mean(1)); wid.append(np.full(len(idx), wi))
    pts = np.vstack(pts); surf = np.concatenate(surf); wid = np.concatenate(wid)
    log(f"{len(data)} wells, cloud {len(pts)} pts; building single KDTree")
    tree = cKDTree(pts)

    allT = []; allFF = []; allBD = []; allD = []
    for si, (wi, X, Y, Z, T, Ti, F, m) in enumerate(data):
        ev = (~np.isfinite(Ti)) & m; pp = np.isfinite(Ti) & m
        if ev.sum() < 20 or pp.sum() < 40:
            continue
        q = np.column_stack([X, Y])
        dd, ii = tree.query(q, k=40)               # over-query, then drop same-well pts
        same = wid[ii] == wi
        dd = np.where(same, np.inf, dd)
        order = np.argsort(dd, axis=1)
        dd = np.take_along_axis(dd, order, 1)[:, :8]
        ii = np.take_along_axis(ii, order, 1)[:, :8]
        wgt = 1.0 / (dd + 1.0); wgt /= wgt.sum(1, keepdims=True)
        imp = (wgt * surf[ii]).sum(1)              # imputed mean-surface at every row
        D = T + Z; last = T[pp][-1]
        b = np.where(pp)[0][-1]; cb = D[b] - imp[b]  # boundary-anchored offset
        tvt_bd = imp[ev] + cb - Z[ev]
        nn = dd[ev, 0]                              # nearest other-well distance per eval row
        allT.append(T[ev]); allFF.append(np.full(ev.sum(), last)); allBD.append(tvt_bd); allD.append(nn)
        if (si + 1) % 150 == 0:
            log(f"  {si+1}/{len(data)}")
    allT = np.concatenate(allT); allFF = np.concatenate(allFF)
    allBD = np.concatenate(allBD); allD = np.concatenate(allD)

    def rmse(p): return np.sqrt(np.mean((allT - p) ** 2))
    print(f"\n=== POOLED over {len(allT)} eval rows, {len(data)} wells ===", flush=True)
    print(f"  ffill:               {rmse(allFF):.3f}", flush=True)
    print(f"  boundary (all rows): {rmse(allBD):.3f}", flush=True)
    for th in [30, 50, 100, 200, 400, 800]:
        g = allD < th; blend = np.where(g, allBD, allFF)
        print(f"  gated<{th:4d}ft: {g.mean()*100:5.1f}% rows -> pooled {rmse(blend):.3f}  (win {rmse(allFF)-rmse(blend):+.3f})", flush=True)
    np.savez("/tmp/rogii_spatial_probe.npz", T=allT, FF=allFF, BD=allBD, D=allD)
    log("saved /tmp/rogii_spatial_probe.npz")


if __name__ == "__main__":
    main()
