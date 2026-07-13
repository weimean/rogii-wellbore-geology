"""Test the 7.016 notebook's CORE idea on our 773-well CV: work in the stratigraphic level
U = TVT + Z (smooth structural surface), fit it on the known prefix, extrapolate, and recover
TVT = U - Z using the KNOWN eval-zone Z. Target-free, per-well, no training -> just pooled RMSE.

Compares against ffill (15.91) and our best (canonical 13.52). If datum projection is competitive,
our 'level is geology-locked' ceiling was wrong (we dismissed U=TVT+Z too early).

    uv run python -u tools/datum_test.py
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from pathlib import Path
from src.data import load_split, TARGET_INPUT
from src.model import baseline_ffill

t0 = time.time()
def log(m): print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


def robust_poly(s, u, deg, iters=3):
    """IRLS Huber-ish robust polyfit."""
    w = np.ones_like(s)
    for _ in range(iters):
        c = np.polyfit(s, u, deg, w=w)
        r = u - np.polyval(c, s)
        sc = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9
        w = 1.0 / np.maximum(1.0, np.abs(r) / (2 * sc))
    return c


def main():
    wells = load_split(Path("data/raw/train"))
    preds = {k: [] for k in ["ffill", "U_const", "U_lin", "U_quad", "U_rquad", "blend75"]}
    tru = []
    for w in wells:
        h = w.horizontal
        if "TVT" not in h.columns:
            continue
        em = w.eval_mask.to_numpy(); kn = h[h[TARGET_INPUT].notna()]
        if em.sum() < 10 or len(kn) < 200:
            continue
        MD = h["MD"].to_numpy(float); Z = h["Z"].to_numpy(float)
        TVTin = h[TARGET_INPUT].to_numpy(float); TVT = h["TVT"].to_numpy(float)
        km = h[TARGET_INPUT].notna().to_numpy()
        s = (MD - MD.min()) / (MD.max() - MD.min() + 1e-9)
        U = TVTin + Z                       # known-zone stratigraphic level
        sk, Uk = s[km], U[km]
        Zev, sev = Z[em], s[em]
        last_U = float(TVTin[km][-1] + Z[km][-1])
        ff = baseline_ffill(h)[em]
        # datum projections (TVT = U_hat - Z_eval)
        p_const = last_U - Zev
        c1 = np.polyfit(sk, Uk, 1); p_lin = np.polyval(c1, sev) - Zev
        c2 = np.polyfit(sk, Uk, 2); p_quad = np.polyval(c2, sev) - Zev
        cr = robust_poly(sk, Uk, 2); p_rquad = np.polyval(cr, sev) - Zev
        blend = 0.25 * ff + 0.75 * p_rquad
        for k, v in [("ffill", ff), ("U_const", p_const), ("U_lin", p_lin),
                     ("U_quad", p_quad), ("U_rquad", p_rquad), ("blend75", blend)]:
            preds[k].append(v)
        tru.append(TVT[em])

    y = np.concatenate(tru)
    print(f"\n=== pooled RMSE over {len(tru)} wells (target-free datum projection) ===", flush=True)
    for k in preds:
        p = np.concatenate(preds[k])
        p = np.where(np.isfinite(p), p, np.nanmedian(p))
        print(f"  {k:10s} {np.sqrt(np.mean((y-p)**2)):.3f}", flush=True)
    print("  (ref: ffill 15.91, our best gbm_stack 13.52)", flush=True)


if __name__ == "__main__":
    main()
