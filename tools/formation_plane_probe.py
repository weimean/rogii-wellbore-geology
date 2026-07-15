"""FormationPlaneKNN (top-team lever): recover per-well TVT datum from a spatial PLANE fit of each
formation surface over neighbour wells. Unlike our crude KNN-average (needs a close twin), a plane
extracts regional dip from k=10 spread-out neighbours -> should work for MOST wells.

Per train well: one reference point (median X, median Y, median depth of each of 6 formations).
For a target well (LOWO, self excluded): find k nearest OTHER wells by median XY, fit IDW WLS plane
depth ~ a*X + b*Y + c per formation, evaluate at the well's rows. Datum b = median(TVT_kn + Z_kn -
surface_kn) over the known prefix. tvtF_f = -Z_ev + surface_ev + b_f. Report pooled RMSE vs ffill.

    uv run python -u tools/formation_plane_probe.py
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
FORMS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]


def plane_fit_eval(nbr_xy, nbr_depth, w, qxy, ctr):
    """IDW-weighted least-squares plane depth ~ a*(X-cx)+b*(Y-cy)+c from neighbours; eval at qxy.
    Coordinates CENTERED at ctr (the query well's median XY) for numerical stability."""
    nx = nbr_xy[:, 0] - ctr[0]; ny = nbr_xy[:, 1] - ctr[1]
    A = np.column_stack([nx, ny, np.ones(len(nbr_xy))])
    try:
        coef = np.linalg.lstsq(A * w[:, None], nbr_depth * w, rcond=None)[0]
    except Exception:
        coef = np.array([0.0, 0.0, np.average(nbr_depth, weights=w)])
    return np.column_stack([qxy[:, 0] - ctr[0], qxy[:, 1] - ctr[1], np.ones(len(qxy))]) @ coef


def main():
    wells = load_split(Path("data/raw/train"))
    med_xy = []; med_depth = []; data = []
    for w in wells:
        h = w.horizontal
        if not all(f in h.columns for f in FORMS):
            continue
        X = h["X"].to_numpy(); Y = h["Y"].to_numpy(); Z = h["Z"].to_numpy()
        T = h["TVT"].to_numpy(); Ti = h["TVT_input"].to_numpy(); F = h[FORMS].to_numpy()
        m = np.isfinite(X) & np.isfinite(Y) & np.isfinite(Z) & np.isfinite(F).all(1)
        if m.sum() < 80:
            continue
        mx, my = np.median(X[m]), np.median(Y[m])
        md = np.array([np.median(F[m, fi]) for fi in range(6)])
        med_xy.append([mx, my]); med_depth.append(md)
        data.append((len(data), X, Y, Z, T, Ti, F, m))
    med_xy = np.array(med_xy); med_depth = np.array(med_depth)  # (Nwell,2),(Nwell,6)
    tree = cKDTree(med_xy)
    log(f"{len(data)} wells")

    # pooled eval RMSE for each formation's tvtF and the 6-mean, vs ffill and oracle(own tops)
    allT = []; allFF = []; allF = {fi: [] for fi in range(6)}; allOr = []
    K = 12
    for wi, X, Y, Z, T, Ti, F, m in data:
        ev = (~np.isfinite(Ti)) & m; pp = np.isfinite(Ti) & m
        if ev.sum() < 20 or pp.sum() < 40:
            continue
        dd, ii = tree.query(med_xy[wi], k=K + 1)      # +1 to drop self
        ii = ii[ii != wi][:K]; dd = dd[:len(ii)] if len(dd) == len(ii) else tree.query(med_xy[wi], k=K + 1)[0][1:K+1]
        wgt = 1.0 / (dd[:K] + 1.0) if len(dd) >= K else 1.0 / (dd + 1.0)
        eidx = np.where(ev)[0]; kidx = np.where(pp)[0]
        qxy_e = np.column_stack([X[eidx], Y[eidx]]); qxy_k = np.column_stack([X[kidx], Y[kidx]])
        last = T[kidx][-1]
        allT.append(T[eidx]); allFF.append(np.full(len(eidx), last))
        tvtF_all = []
        for fi in range(6):
            surf_e = plane_fit_eval(med_xy[ii], med_depth[ii, fi], wgt[:len(ii)], qxy_e, med_xy[wi])
            surf_k = plane_fit_eval(med_xy[ii], med_depth[ii, fi], wgt[:len(ii)], qxy_k, med_xy[wi])
            b = np.median(T[kidx] + Z[kidx] - surf_k)
            tvtF = -Z[eidx] + surf_e + b
            allF[fi].append(tvtF); tvtF_all.append(tvtF)
        # oracle: own true formation surface (upper bound)
        b_or = np.median(T[kidx] + Z[kidx] - F[kidx].mean(1))
        allOr.append(-Z[eidx] + F[eidx].mean(1) + b_or)
    allT = np.concatenate(allT); allFF = np.concatenate(allFF); allOr = np.concatenate(allOr)
    def rmse(p): return np.sqrt(np.mean((allT - p) ** 2))
    print(f"\n=== pooled eval RMSE, {len(allT)} rows ===", flush=True)
    print(f"  ffill:                 {rmse(allFF):.3f}", flush=True)
    for fi in range(6):
        pf = np.concatenate(allF[fi])
        print(f"  tvtF_{FORMS[fi]:6s} (plane):  {rmse(pf):.3f}", flush=True)
    mean6 = np.mean([np.concatenate(allF[fi]) for fi in range(6)], axis=0)
    print(f"  tvtF mean-of-6:        {rmse(mean6):.3f}", flush=True)
    print(f"  ORACLE own tops:       {rmse(allOr):.3f}  (upper bound)", flush=True)


if __name__ == "__main__":
    main()
