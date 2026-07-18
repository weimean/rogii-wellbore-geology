"""Neighbor TVT-curve transfer (notes/19, the ROGII webinar's precise method): for each eval row, take
the SINGLE nearest OTHER-well point's datum D=TVT+Z (which carries the neighbour's local WIGGLE, unlike a
smooth surface fit), calibrate the per-well offset on the known zone, predict TVT = D_nn - Z + c. Test on
CLOSE-neighbour wells whether this beats ffill AND the smooth mean-surface (i.e. captures the wiggle).

    uv run python -u tools/neighbor_transfer_probe.py
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
TI = "TVT_input"
FORMS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]


def main():
    wells = load_split(Path("data/raw/train"))
    # dense cloud: (X,Y) -> D=TVT+Z and mean-surface, per point, tagged by well
    pts = []; Dv = []; Sv = []; wid = []; data = []
    for k, w in enumerate(wells):
        h = w.horizontal
        X = h["X"].to_numpy(float); Y = h["Y"].to_numpy(float); Z = h["Z"].to_numpy(float)
        T = h["TVT"].to_numpy(float) if "TVT" in h.columns else np.full(len(X), np.nan)
        Ti = h[TI].to_numpy(float)
        okf = all(f in h.columns for f in FORMS)
        S = h[FORMS].to_numpy(float).mean(1) if okf else np.full(len(X), np.nan)
        m = np.isfinite(X) & np.isfinite(Y) & np.isfinite(Z) & np.isfinite(T)
        if m.sum() < 80:
            continue
        data.append((k, X, Y, Z, T, Ti, S, m))
        idx = np.where(m)[0][::3]
        pts.append(np.column_stack([X[idx], Y[idx]])); Dv.append((T + Z)[idx]); Sv.append(S[idx]); wid.append(np.full(len(idx), k))
    pts = np.vstack(pts); Dv = np.concatenate(Dv); Sv = np.concatenate(Sv); wid = np.concatenate(wid)
    tree = cKDTree(pts); log(f"{len(data)} wells, cloud {len(pts)}")

    allT = []; allFF = []; allNN = []; allSURF = []; allD = []
    for k, X, Y, Z, T, Ti, S, m in data:
        ev = (~np.isfinite(Ti)) & m; pp = np.isfinite(Ti) & m
        if ev.sum() < 20 or pp.sum() < 40:
            continue
        eidx = np.where(ev)[0]; kidx = np.where(pp)[0]
        q_e = np.column_stack([X[eidx], Y[eidx]]); q_k = np.column_stack([X[kidx], Y[kidx]])
        # k=1 nearest OTHER-well point (over-query, drop self)
        def nn(q):
            dd, ii = tree.query(q, k=30)
            same = wid[ii] == k; dd = np.where(same, np.inf, dd)
            j = np.argmin(dd, 1)
            sel = ii[np.arange(len(q)), j]; dsel = dd[np.arange(len(q)), j]
            return sel, dsel
        se, de = nn(q_e); sk, _ = nn(q_k)
        last = T[kidx][-1]
        # neighbour-D transfer (wiggle-preserving), calibrated on known zone
        c = np.median(T[kidx] - (Dv[sk] - Z[kidx]))
        pred_nn = Dv[se] - Z[eidx] + c
        # smooth mean-surface reconstruction (our formation-plane style), calibrated the same way
        c2 = np.median(T[kidx] + Z[kidx] - Sv[sk])
        pred_surf = -Z[eidx] + Sv[se] + c2 if np.isfinite(Sv[se]).all() else np.full(len(eidx), last)
        allT.append(T[eidx]); allFF.append(np.full(len(eidx), last))
        allNN.append(pred_nn); allSURF.append(pred_surf); allD.append(de)
    allT = np.concatenate(allT); allFF = np.concatenate(allFF)
    allNN = np.concatenate(allNN); allSURF = np.concatenate(allSURF); allD = np.concatenate(allD)
    def rmse(p, msk): return np.sqrt(np.mean((allT[msk] - p[msk]) ** 2))
    print(f"\n=== neighbour TVT-curve transfer, gated by nearest-neighbour distance ===", flush=True)
    for th in (30, 80, 200, 1e9):
        g = allD < th
        if g.sum() < 200:
            continue
        lab = f"<{th:g}ft" if th < 1e8 else "ALL"
        print(f"  {lab:>7} ({g.mean()*100:4.1f}% rows): ffill {rmse(allFF,g):6.2f} | smooth-surf {rmse(allSURF,g):6.2f} | NN-transfer {rmse(allNN,g):6.2f}", flush=True)


if __name__ == "__main__":
    main()
