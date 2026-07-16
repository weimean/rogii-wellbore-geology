"""Validate anisotropic kriging of the per-well level (notes/15). L_w = per-well mean(TVT_eval -
ffill_last) = the level offset our anchor struggles to predict (R^2~0.10). Krige L_w LOWO from
neighbour wells with a fitted anisotropic gaussian variogram; compare to ffill, oracle-level,
isotropic KNN, and our Ridge anchor. If kriging predicts L_w well, it's a level lever.

    uv run python -u tools/krige_datum_probe.py
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
from src.data import load_split
from src.model import fit_level_model, level_regression

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)
TI = "TVT_input"


def main():
    wells = load_split(Path("data/raw/train"))
    mx = []; my = []; Lw = []; data = []
    for w in wells:
        h = w.horizontal; em = w.eval_mask.to_numpy()
        if em.sum() < 10 or "TVT" not in h.columns:
            continue
        X = h["X"].to_numpy(float); Y = h["Y"].to_numpy(float)
        m = np.isfinite(X) & np.isfinite(Y)
        last = float(h[h[TI].notna()][TI].iloc[-1])
        tr = h["TVT"].to_numpy()[em]
        mx.append(np.median(X[m])); my.append(np.median(Y[m]))
        Lw.append(float(np.mean(tr) - last)); data.append((w, em, last, tr))
    mx = np.array(mx); my = np.array(my); Lw = np.array(Lw)
    N = len(Lw); log(f"{N} wells; L_w std {Lw.std():.2f} ft")

    # anchor (Ridge) per-well level for comparison
    anchor_L = np.full(N, np.nan)
    lm = fit_level_model(wells)
    for i, (w, em, last, tr) in enumerate(data):
        anchor_L[i] = float(np.mean(level_regression(w, lm)[em]) - last)

    # anisotropic krige of L_w, LOWO. params seed from notebook, then a small local sweep.
    def krige(TH, ANISO, RNG, K):
        c, s = np.cos(TH), np.sin(TH)
        u = mx * c + my * s; v = (-mx * s + my * c) / ANISO   # rotate + squash minor axis
        pts = np.column_stack([u, v]); tree = cKDTree(pts)
        pred = np.zeros(N)
        dd, ii = tree.query(pts, k=min(K + 1, N))
        for i in range(N):
            j = ii[i][ii[i] != i][:K]; d = np.linalg.norm(pts[j] - pts[i], axis=1)
            wt = np.exp(-0.5 * (d / RNG) ** 2)
            pred[i] = np.sum(wt * Lw[j]) / max(wt.sum(), 1e-9) if wt.sum() > 0 else 0.0
        return pred

    truth = np.concatenate([tr for _, _, _, tr in data])
    lastv = np.concatenate([np.full(em.sum(), last) for _, em, last, _ in data])
    counts = np.array([em.sum() for _, em, _, _ in data])
    def pooled(Lpred):
        pw = np.repeat(lastv[np.cumsum(counts) - counts] * 0, 1)  # placeholder
        p = np.concatenate([np.full(c, last + Lp) for (_, em, last, _), c, Lp in zip(data, counts, Lpred)])
        return np.sqrt(np.mean((truth - p) ** 2))

    print(f"\n=== per-well level L_w prediction (LOWO) ===", flush=True)
    print(f"  ffill (L=0)         pooled {pooled(np.zeros(N)):.3f}   |L_w err| RMSE {np.sqrt(np.mean(Lw**2)):.2f}", flush=True)
    print(f"  ORACLE level        pooled {pooled(Lw):.3f}", flush=True)
    va = np.isfinite(anchor_L)
    print(f"  Ridge anchor        pooled {pooled(np.where(va,anchor_L,0)):.3f}   L_w err {np.sqrt(np.mean((Lw[va]-anchor_L[va])**2)):.2f}", flush=True)
    kiso = krige(0.0, 1.0, 500.0, 25)
    print(f"  isotropic krige     pooled {pooled(kiso):.3f}   L_w err {np.sqrt(np.mean((Lw-kiso)**2)):.2f}", flush=True)
    print("\n  anisotropic sweep (TH, ANISO, RNG, K) -> L_w err RMSE / pooled:", flush=True)
    best = None
    for TH in (0.0, 0.79, 1.57, 2.2166):
        for ANISO in (1.0, 2.0, 3.0):
            for RNG in (300.0, 500.0, 1000.0):
                kp = krige(TH, ANISO, RNG, 25)
                e = np.sqrt(np.mean((Lw - kp) ** 2))
                if best is None or e < best[0]:
                    best = (e, TH, ANISO, RNG, pooled(kp))
    print(f"  BEST aniso: err {best[0]:.2f}  pooled {best[4]:.3f}  (TH={best[1]}, ANISO={best[2]}, RNG={best[3]})", flush=True)


if __name__ == "__main__":
    main()
