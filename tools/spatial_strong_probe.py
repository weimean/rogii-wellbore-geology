"""Strengthen the offset-well imputation toward the 0.01 oracle. Grouped CV understates spatial
features (test wells have neighbours, train wells mostly don't), so we optimise IMPUTATION FIDELITY
on the CLOSE-NEIGHBOUR subset of train wells (test-like conditions) instead.

Compares, on wells with a neighbour < gate ft (self-excluded):
  ffill | mean-surf drift (current) | ridge-6-surfaces | neighbour-D-direct | oracle(true surf)

    uv run python -u tools/spatial_strong_probe.py
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
from sklearn.linear_model import Ridge
from src.data import load_split

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)
forms = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]


def main():
    wells = load_split(Path("data/raw/train"))
    pts = []; S6 = []; DD = []; wid = []; data = []
    for k, w in enumerate(wells):
        h = w.horizontal
        if not all(f in h.columns for f in forms):
            continue
        X = h["X"].to_numpy(); Y = h["Y"].to_numpy(); Z = h["Z"].to_numpy()
        T = h["TVT"].to_numpy(); Ti = h["TVT_input"].to_numpy(); F = h[forms].to_numpy()
        m = np.isfinite(X) & np.isfinite(Y) & np.isfinite(Z) & np.isfinite(T) & np.isfinite(F).all(1)
        if m.sum() < 80:
            continue
        data.append((k, X, Y, Z, T, Ti, F, m))
        idx = np.where(m)[0][::5]
        pts.append(np.column_stack([X[idx], Y[idx]])); S6.append(F[idx]); DD.append((T + Z)[idx])
        wid.append(np.full(len(idx), k))
    pts = np.vstack(pts); S6 = np.vstack(S6); DD = np.concatenate(DD); wid = np.concatenate(wid)
    tree = cKDTree(pts)
    log(f"{len(data)} wells; cloud {len(pts)} pts")

    def knn(qpts, self_k, k=8, kq=300):
        dd, ii = tree.query(np.nan_to_num(qpts, nan=1e9), k=kq)   # deep query so a close neighbour
        dd = np.where(wid[ii] == self_k, np.inf, dd)              # survives dense self-point masking
        o = np.argsort(dd, 1)
        dd = np.take_along_axis(dd, o, 1)[:, :k]; ii = np.take_along_axis(ii, o, 1)[:, :k]
        w = 1.0 / (dd + 1.0); ws = w.sum(1, keepdims=True)
        wgt = np.where(ws > 0, w / np.where(ws > 0, ws, 1.0), 0.0)
        return dd, ii, wgt

    GATES = [50, 150, 500]
    acc = {g: {n: [] for n in ["ff", "cur", "ridge", "nD", "oracle", "cnt"]} for g in GATES}
    for (k, X, Y, Z, T, Ti, F, m) in data:
        ev = (~np.isfinite(Ti)) & m; pp = np.isfinite(Ti) & m
        if ev.sum() < 20 or pp.sum() < 40:
            continue
        eidx = np.where(ev)[0]; b = np.where(pp)[0][-1]
        qrows = np.append(eidx, np.append(np.where(pp)[0], b))  # eval + prePS + boundary
        qpts = np.column_stack([X[qrows], Y[qrows]])
        dd, ii, wgt = knn(qpts, k)
        n_ev = len(eidx); n_pp = pp.sum()
        surf6 = np.einsum('qk,qkf->qf', wgt, S6[ii])            # imputed 6 surfaces at qrows
        dimp = (wgt * DD[ii]).sum(1)                            # imputed D=TVT+Z at qrows
        se, sp = surf6[:n_ev], surf6[n_ev:n_ev + n_pp]; sb = surf6[-1]
        de, dp = dimp[:n_ev], dimp[n_ev:n_ev + n_pp]; db = dimp[-1]
        Dtrue = T + Z; last = T[b]
        nn_ev = dd[:n_ev, 0]; nn_med = np.median(nn_ev)
        # predictions on eval
        cur = se.mean(1) + (Dtrue[b] - sb.mean()) - Z[eidx]                 # current mean-surf drift
        try:
            rg = Ridge(alpha=1e3).fit(sp, Dtrue[np.where(pp)[0]]); cD = rg.predict(se)
        except Exception:
            cD = se.mean(1)
        ridge = cD - Z[eidx]
        cN = np.mean(Dtrue[np.where(pp)[0]] - dp); nD = de + cN - Z[eidx]   # neighbour-D direct, offset-cal
        orc = F[eidx].mean(1) + (Dtrue[b] - F[b].mean()) - Z[eidx]          # oracle (true surf)
        tt = T[eidx]
        def r(p):
            p = np.where(np.isfinite(p), p, last)   # nan rows (no neighbour) -> ffill fallback
            return np.sqrt(np.mean((tt - p) ** 2))
        for g in GATES:
            if nn_med < g:
                acc[g]["ff"].append(r(np.full(n_ev, last))); acc[g]["cur"].append(r(cur))
                acc[g]["ridge"].append(r(ridge)); acc[g]["nD"].append(r(nD))
                acc[g]["oracle"].append(r(orc)); acc[g]["cnt"].append(1)
    print("\n=== close-neighbour subset (per-well RMSE means) ===", flush=True)
    print(f"{'gate':>6} {'n':>4} {'ffill':>7} {'cur':>7} {'ridge':>7} {'neighD':>7} {'oracle':>7}", flush=True)
    for g in GATES:
        a = acc[g]; n = len(a["cnt"])
        if n:
            print(f"{g:>6} {n:>4} {np.mean(a['ff']):>7.2f} {np.mean(a['cur']):>7.2f} "
                  f"{np.mean(a['ridge']):>7.2f} {np.mean(a['nD']):>7.2f} {np.mean(a['oracle']):>7.2f}", flush=True)


if __name__ == "__main__":
    main()
