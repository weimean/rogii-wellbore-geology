"""Clean strat-based SELF-correlation (notes/16, the ROGII webinar's "precise" method): correlate the
horizontal GR against the well's OWN high-res known-zone GR(TVT) — NOT the smoother typewell. Per eval
block, scan the datum U (TVT = U - Z, Z known) to maximise the NORMALISED cross-correlation (scale/offset
invariant -> no calibration) between the eval-GR block shape and GR_known(U - Z). Bimodal-hedged.
Compare pooled RMSE vs ffill, and vs the SAME scan against the TYPEWELL (to isolate self- vs typewell-ref).

    uv run python -u tools/self_corr_probe.py
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


def _ncc_scan(ge, z, ref_tvt, ref_gr, last_u, block, u_range, n_u, bimodal):
    """Block-wise NCC datum scan of eval GR against a (TVT->GR) reference. Returns TVT per eval row."""
    n = len(ge); pred = np.full(n, np.nan)
    us = np.linspace(last_u - u_range, last_u + u_range, n_u)
    tlo, thi = ref_tvt.min(), ref_tvt.max()
    for a in range(0, n, block):
        b = min(a + block, n)
        gb = ge[a:b]; zb = z[a:b]
        if gb.std() < 1e-6:
            continue
        gb0 = (gb - gb.mean()) / (gb.std() + 1e-9)
        nccs = np.full(len(us), -2.0)
        for j, u in enumerate(us):
            tv = u - zb
            if (tv.min() < tlo - 5) or (tv.max() > thi + 5):
                continue
            kk = np.interp(tv, ref_tvt, ref_gr)
            if kk.std() < 1e-6:
                continue
            kk0 = (kk - kk.mean()) / (kk.std() + 1e-9)
            nccs[j] = float(np.mean(gb0 * kk0))
        if nccs.max() <= -2:
            continue
        if bimodal:
            w = np.exp((nccs - nccs.max()) / 0.05); w /= w.sum()
            u_hat = float((w * us).sum())
        else:
            u_hat = float(us[np.argmax(nccs)])
        pred[a:b] = u_hat - zb
    return pred


def self_corr(well, ref="self", block=80, u_range=60.0, n_u=201, bimodal=True):
    h, t = well.horizontal, well.typewell
    kn = h[h[TI].notna()]; ev = h[h[TI].isna()]
    if len(ev) < 20 or len(kn) < 80:
        return None
    if ref == "self":
        rt = kn[TI].to_numpy(float); rg = kn["GR"].to_numpy(float)
    else:  # typewell
        rt = t["TVT"].to_numpy(float); rg = t["GR"].ffill().bfill().to_numpy(float)
    m = np.isfinite(rt) & np.isfinite(rg); rt, rg = rt[m], rg[m]
    if len(rt) < 40:
        return None
    o = np.argsort(rt); rt = rt[o]; rg = gaussian_filter1d(rg[o], 2)
    ei = ev.index.values
    ge = gaussian_filter1d(h["GR"].interpolate(limit_direction="both").fillna(np.nanmean(rg)).to_numpy(float), 2)[ei]
    z = ev["Z"].to_numpy(float)
    last_u = float(kn[TI].iloc[-1] + kn["Z"].iloc[-1])
    pred = _ncc_scan(ge, z, rt, rg, last_u, block, u_range, n_u, bimodal)
    return np.where(np.isfinite(pred), pred, last_u - z)


def main():
    wells = load_split(Path("data/raw/train"))
    T = []; FF = []; SELF = []; TW = []
    for w in wells:
        h = w.horizontal; em = w.eval_mask.to_numpy()
        if em.sum() < 20 or "TVT" not in h.columns:
            continue
        ps = self_corr(w, ref="self"); pt = self_corr(w, ref="tw")
        if ps is None or pt is None:
            continue
        T.append(h["TVT"].to_numpy()[em]); FF.append(np.full(em.sum(), float(h[h[TI].notna()][TI].iloc[-1])))
        SELF.append(ps); TW.append(pt)
    T = np.concatenate(T)
    def r(x): return np.sqrt(np.mean((T - np.concatenate(x)) ** 2))
    print(f"\n=== self-correlation, pooled over {len(T)} eval rows ({len(FF)} wells) ===", flush=True)
    print(f"  ffill:                    {r(FF):.3f}", flush=True)
    print(f"  block-NCC vs TYPEWELL:    {r(TW):.3f}", flush=True)
    print(f"  block-NCC vs SELF (own):  {r(SELF):.3f}", flush=True)


if __name__ == "__main__":
    main()
