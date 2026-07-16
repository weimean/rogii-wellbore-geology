"""Validate the forum's key lever (notes/15): affine-calibrated GR datum scan. TVT_i = U - Z_i
(U = per-well datum, Z known); scan U, match horizontal GR (normalised to typewell scale via a
heel-fit affine gain/offset) against typewell GR(TVT). Georgy: raw scan localises 8%, heel-calibrated
~80%. Test the DATUM (constant-U) scan on train vs truth, with/without affine cal + bimodal hedge.

    uv run python -u tools/gr_datum_probe.py
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from src.data import load_split

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)
TI = "TVT_input"


def gr_scan(well, affine=True, u_range=70.0, n_u=281, huber=4.0, bimodal=True):
    h, t = well.horizontal, well.typewell
    kn = h[h[TI].notna()]; ev = h[h[TI].isna()]
    if len(ev) == 0 or len(kn) < 50:
        return None
    tw_tvt = t["TVT"].to_numpy(float); tw_gr = t["GR"].ffill().bfill().to_numpy(float)
    o = np.argsort(tw_tvt); tw_tvt = tw_tvt[o]; tw_gr = gaussian_filter1d(tw_gr[o], 3)
    kt = kn[TI].to_numpy(float); kg = kn["GR"].to_numpy(float)
    m = np.isfinite(kt) & np.isfinite(kg)
    if m.sum() < 30:
        return None
    tw_at_kn = np.interp(kt[m], tw_tvt, tw_gr)
    if affine:
        a, b = np.polyfit(tw_at_kn, kg[m], 1)     # kg ~ a*tw_gr + b
        if not np.isfinite(a) or abs(a) < 1e-6:
            a, b = 1.0, 0.0
    else:
        a, b = 1.0, 0.0
    resid = kg[m] - (a * tw_at_kn + b)
    s = float(np.clip(np.std(resid), 5.0, 60.0))
    ei = ev.index.values
    gr_e = h["GR"].interpolate(limit_direction="both").fillna(np.nanmean(tw_gr)).to_numpy(float)[ei]
    g_norm = (gr_e - b) / a                          # horizontal GR -> typewell scale
    z_e = ev["Z"].to_numpy(float)
    last_u = float(kn[TI].iloc[-1] + kn["Z"].iloc[-1])
    us = np.linspace(last_u - u_range, last_u + u_range, n_u)
    # cost(U) = mean clipped-square GR misfit for TVT = U - Z
    tvt_grid = us[:, None] - z_e[None, :]            # (n_u, n_ev)
    pred_gr = np.interp(tvt_grid.ravel(), tw_tvt, tw_gr).reshape(tvt_grid.shape)
    r = np.clip((g_norm[None, :] - pred_gr) / s, -huber, huber)
    cost = (r ** 2).mean(1)
    if bimodal:
        wgt = np.exp(-(cost - cost.min()) / 0.5); wgt /= wgt.sum()
        u_hat = float((wgt * us).sum())
    else:
        u_hat = float(us[np.argmin(cost)])
    return (u_hat - z_e).astype(float), float(cost.min())


def main():
    wells = load_split(Path("data/raw/train"))
    variants = {"ffill": [], "scan_noaff": [], "scan_aff": [], "scan_aff_bi": []}
    truth = []
    for w in wells:
        h = w.horizontal; em = w.eval_mask.to_numpy()
        if em.sum() < 10 or "TVT" not in h.columns:
            continue
        tr = h["TVT"].to_numpy()[em]; last = float(h[h[TI].notna()][TI].iloc[-1])
        r0 = gr_scan(w, affine=False, bimodal=False)
        r1 = gr_scan(w, affine=True, bimodal=False)
        r2 = gr_scan(w, affine=True, bimodal=True)
        if r0 is None or r1 is None or r2 is None:
            continue
        truth.append(tr); variants["ffill"].append(np.full(em.sum(), last))
        variants["scan_noaff"].append(r0[0]); variants["scan_aff"].append(r1[0]); variants["scan_aff_bi"].append(r2[0])
    truth = np.concatenate(truth)
    def rmse(k): return np.sqrt(np.mean((truth - np.concatenate(variants[k])) ** 2))
    print(f"\n=== GR datum scan, pooled over {len(truth)} eval rows ===", flush=True)
    print(f"  ffill:                 {rmse('ffill'):.3f}", flush=True)
    print(f"  GR-scan (no affine):   {rmse('scan_noaff'):.3f}", flush=True)
    print(f"  GR-scan (affine cal):  {rmse('scan_aff'):.3f}", flush=True)
    print(f"  GR-scan (affine+bimod):{rmse('scan_aff_bi'):.3f}", flush=True)


if __name__ == "__main__":
    main()
