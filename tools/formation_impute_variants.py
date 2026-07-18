"""Formation-surface imputation VARIANTS, bucketed by nearest-neighbour distance (notes/21).

The formation surfaces are a near-perfect TVT signal (oracle 0.006) but the plane imputation is only
good where a NEIGHBOUR is close. On train 96% of wells are isolated (median nn 356ft) so the POOLED
number (88) is meaningless — the scored TEST wells are offset wells with CLOSE neighbours. So the lever's
value lives in the near-distance buckets. This probe compares surface models (linear plane / quadratic /
IDW / universal-kriging) x k, reporting mean-of-6 tvtF RMSE PER nn-distance bucket vs ffill and oracle.
The winner in the <30/<80/<200ft buckets is what improves public.

    uv run python -u tools/formation_impute_variants.py
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


def _plane(nbr_xy, depth, w, qxy, ctr, quad=False):
    nx = nbr_xy[:, 0] - ctr[0]; ny = nbr_xy[:, 1] - ctr[1]
    if quad:
        A = np.column_stack([nx, ny, nx * nx, ny * ny, nx * ny, np.ones(len(nx))])
        Q = np.column_stack([qxy[:, 0] - ctr[0], qxy[:, 1] - ctr[1],
                             (qxy[:, 0] - ctr[0]) ** 2, (qxy[:, 1] - ctr[1]) ** 2,
                             (qxy[:, 0] - ctr[0]) * (qxy[:, 1] - ctr[1]), np.ones(len(qxy))])
    else:
        A = np.column_stack([nx, ny, np.ones(len(nx))])
        Q = np.column_stack([qxy[:, 0] - ctr[0], qxy[:, 1] - ctr[1], np.ones(len(qxy))])
    try:
        coef = np.linalg.lstsq(A * w[:, None], depth * w, rcond=None)[0]
    except Exception:
        return np.full(len(qxy), np.average(depth, weights=w)), None
    return Q @ coef, (A, coef, nx, ny)


def _idw(nbr_xy, depth, qxy, p=2.0):
    """Inverse-distance-weighted average at each query point."""
    d = np.sqrt(((qxy[:, None, :] - nbr_xy[None, :, :]) ** 2).sum(2)) + 1.0   # (nq, k)
    w = 1.0 / d ** p
    return (w * depth[None, :]).sum(1) / w.sum(1)


def _uk(nbr_xy, depth, w, qxy, ctr):
    """Universal kriging (approx): linear plane trend + IDW of the plane residuals."""
    surf, extra = _plane(nbr_xy, depth, w, qxy, ctr, quad=False)
    if extra is None:
        return surf
    A, coef, nx, ny = extra
    resid = depth - A @ coef
    return surf + _idw(nbr_xy, resid, qxy, p=2.0)


VARIANTS = [
    ("plane_k12", dict(kind="plane", k=12)),
    ("plane_k40", dict(kind="plane", k=40)),
    ("uk_k24", dict(kind="uk", k=24)),
    ("uk_k40", dict(kind="uk", k=40)),
    ("uk_k60", dict(kind="uk", k=60)),
    ("uk_k80", dict(kind="uk", k=80)),
    ("twin_uk40", dict(kind="twin", k=40)),   # nearest-twin override <25ft, else uk_k40
]


def impute_mean6(cfg, med_xy, med_depth, ii, wi, qxy_e, qxy_k, T, Z, kidx, eidx):
    """mean-of-6 tvtF on eval rows for one variant. datum b calibrated on known zone."""
    ctr = med_xy[wi]; nbr_xy = med_xy[ii]
    dd = np.sqrt(((nbr_xy - ctr) ** 2).sum(1)); w = 1.0 / (dd + 1.0)
    tvtF = []
    for fi in range(6):
        depth = med_depth[ii, fi]
        if cfg["kind"] == "plane":
            se, _ = _plane(nbr_xy, depth, w, qxy_e, ctr); sk, _ = _plane(nbr_xy, depth, w, qxy_k, ctr)
        elif cfg["kind"] == "quad":
            se, _ = _plane(nbr_xy, depth, w, qxy_e, ctr, quad=True); sk, _ = _plane(nbr_xy, depth, w, qxy_k, ctr, quad=True)
        elif cfg["kind"] == "idw":
            se = _idw(nbr_xy, depth, qxy_e); sk = _idw(nbr_xy, depth, qxy_k)
        elif cfg["kind"] == "twin":
            se = _uk(nbr_xy, depth, w, qxy_e, ctr); sk = _uk(nbr_xy, depth, w, qxy_k, ctr)
            if dd[0] < 25.0:            # a genuine close twin: use its surface directly
                se = np.full(len(qxy_e), depth[0]); sk = np.full(len(qxy_k), depth[0])
        else:  # uk
            se = _uk(nbr_xy, depth, w, qxy_e, ctr); sk = _uk(nbr_xy, depth, w, qxy_k, ctr)
        b = np.median(T[kidx] + Z[kidx] - sk)
        tvtF.append(-Z[eidx] + se + b)
    return np.mean(tvtF, axis=0)


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
        med_xy.append([np.median(X[m]), np.median(Y[m])])
        med_depth.append([np.median(F[m, fi]) for fi in range(6)])
        data.append((len(data), X, Y, Z, T, Ti, F, m))
    med_xy = np.array(med_xy); med_depth = np.array(med_depth)
    tree = cKDTree(med_xy); log(f"{len(data)} wells")

    KMAX = 41
    acc = {name: [] for name, _ in VARIANTS}
    aT = []; aFF = []; aOr = []; aND = []
    for wi, X, Y, Z, T, Ti, F, m in data:
        ev = (~np.isfinite(Ti)) & m; pp = np.isfinite(Ti) & m
        if ev.sum() < 20 or pp.sum() < 40:
            continue
        dd, ii = tree.query(med_xy[wi], k=KMAX + 1)
        keep = ii != wi; ii = ii[keep][:KMAX]; dd = dd[keep][:KMAX]
        nn = dd[0]                                  # nearest OTHER-well distance (the gate)
        eidx = np.where(ev)[0]; kidx = np.where(pp)[0]
        qxy_e = np.column_stack([X[eidx], Y[eidx]]); qxy_k = np.column_stack([X[kidx], Y[kidx]])
        last = T[kidx][-1]
        aT.append(T[eidx]); aFF.append(np.full(len(eidx), last)); aND.append(np.full(len(eidx), nn))
        b_or = np.median(T[kidx] + Z[kidx] - F[kidx].mean(1))
        aOr.append(-Z[eidx] + F[eidx].mean(1) + b_or)
        for name, cfg in VARIANTS:
            k = min(cfg["k"], len(ii))
            pred = impute_mean6(cfg, med_xy, med_depth, ii[:k], wi, qxy_e, qxy_k, T, Z, kidx, eidx)
            acc[name].append(pred)
    aT = np.concatenate(aT); aFF = np.concatenate(aFF); aOr = np.concatenate(aOr); aND = np.concatenate(aND)
    for name in acc:
        acc[name] = np.concatenate(acc[name])

    def rmse(p, msk): return np.sqrt(np.mean((aT[msk] - p[msk]) ** 2))
    buckets = [(0, 30), (30, 80), (80, 200), (200, 500), (500, 1e9)]
    print(f"\n=== mean-of-6 tvtF RMSE by nearest-neighbour distance bucket ({len(aT)} rows) ===", flush=True)
    hdr = "  bucket        rows%   ffill  oracle | " + "  ".join(f"{n:>9s}" for n, _ in VARIANTS)
    print(hdr, flush=True)
    for lo, hi in buckets:
        g = (aND >= lo) & (aND < hi)
        if g.sum() < 100:
            continue
        lab = f"{lo:>4g}-{hi:<5g}" if hi < 1e8 else f"{lo:>4g}+    "
        row = f"  {lab}  {g.mean()*100:5.1f}  {rmse(aFF,g):6.2f}  {rmse(aOr,g):6.3f} | "
        row += "  ".join(f"{rmse(acc[n],g):9.2f}" for n, _ in VARIANTS)
        print(row, flush=True)
    # overall on the near set (<200ft) — the test-relevant regime
    g = aND < 200
    print(f"\n  <200ft near-set ({g.mean()*100:.1f}% of rows): ffill {rmse(aFF,g):.2f}  oracle {rmse(aOr,g):.3f}", flush=True)
    best = sorted(((rmse(acc[n], g), n) for n, _ in VARIANTS))
    for r, n in best:
        print(f"    {n:>10s}: {r:6.2f}", flush=True)


if __name__ == "__main__":
    main()
