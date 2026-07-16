"""Refine the per-well LEVEL estimator (dominant error term). Match eval horizontal GR to the
typewell GR profile near last_known to pin absolute TVT. Report both pooled RMSE and level quality
(estimated c vs true eval-mean). ffill=15.65 (150w) / oracle-level=8.65.

Variants:
  grmean   : match mean eval GR  (baseline that already beat ffill)
  quant    : match a set of quantiles of the eval GR distribution to the typewell's over a band
  hedge    : nearest-to-last among low-cost minima vs softmax
  window R, temp sweeps

    uv run python -u tools/gr_level_probe.py [NWELLS]
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from src.data import load_split

TI = "TVT_input"


def prep(w):
    h, t = w.horizontal, w.typewell
    em = w.eval_mask.to_numpy()
    if em.sum() < 20 or "TVT" not in h.columns:
        return None
    kn = ~em
    tw_tvt = t["TVT"].to_numpy(float); tw_gr = t["GR"].ffill().bfill().to_numpy(float)
    o = np.argsort(tw_tvt); tw_tvt = tw_tvt[o]; tw_gr = tw_gr[o]
    tw_grs = gaussian_filter1d(tw_gr, 2)
    tvt = h["TVT"].to_numpy(float)
    gr = h["GR"].interpolate(limit_direction="both").to_numpy(float)
    ktvt = tvt[kn]; kgr = gr[kn]; last = ktvt[-1]
    tw_at_kn = np.interp(ktvt, tw_tvt, tw_grs)
    mm = np.isfinite(tw_at_kn) & np.isfinite(kgr)
    if mm.sum() < 30:
        return None
    a, b = np.polyfit(tw_at_kn[mm], kgr[mm], 1)
    if not np.isfinite(a) or abs(a) < 1e-6:
        a, b = 1.0, 0.0
    gn = (gr[em] - b) / a
    resid = kgr[mm] - (a * tw_at_kn[mm] + b)
    sigma = float(np.clip(np.std(resid), 3.0, 40.0))
    return dict(tw_tvt=tw_tvt, tw_gr=tw_grs, gn=gn, last=last,
                truth=tvt[em], truemean=float(np.mean(tvt[em])), sigma=sigma, n=int(em.sum()))


def level_grmean(P, R=35.0, temp=1.0, npts=141):
    cs = np.linspace(P["last"] - R, P["last"] + R, npts)
    pg = np.interp(cs, P["tw_tvt"], P["tw_gr"])
    cost = (pg - np.nanmean(P["gn"])) ** 2 / (2 * P["sigma"] ** 2)
    w = np.exp(-(cost - cost.min()) / temp); w /= w.sum()
    return float((w * cs).sum())


def level_quant(P, R=35.0, band=8.0, temp=1.0, npts=141, qs=(0.25, 0.5, 0.75)):
    """For each candidate center c, compare quantiles of eval GR to typewell GR over [c-band,c+band]."""
    tw_tvt, tw_gr = P["tw_tvt"], P["tw_gr"]
    cs = np.linspace(P["last"] - R, P["last"] + R, npts)
    eq = np.nanquantile(P["gn"], qs)
    cost = np.empty(len(cs))
    for i, c in enumerate(cs):
        sel = (tw_tvt >= c - band) & (tw_tvt <= c + band)
        if sel.sum() < 3:
            cost[i] = 1e9; continue
        tq = np.quantile(tw_gr[sel], qs)
        cost[i] = np.mean((eq - tq) ** 2)
    cost = cost / (2 * P["sigma"] ** 2)
    w = np.exp(-(cost - cost.min()) / temp); w /= w.sum()
    return float((w * cs).sum())


def main():
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    wells = load_split(Path("data/raw/train"), limit=nw)
    Ps = [P for P in (prep(w) for w in wells) if P is not None]
    truth = np.concatenate([P["truth"] for P in Ps])
    last_arr = np.array([P["last"] for P in Ps])
    tmean = np.array([P["truemean"] for P in Ps])

    def pooled(levels):
        pred = np.concatenate([np.full(P["n"], lv) for P, lv in zip(Ps, levels)])
        return np.sqrt(np.mean((truth - pred) ** 2))

    def levelrmse(levels):
        return np.sqrt(np.mean((np.array(levels) - tmean) ** 2))

    print(f"=== {len(Ps)} wells, {len(truth)} eval rows ===", flush=True)
    print(f"  ffill              pooled {pooled(last_arr):.3f}  levelRMSE {levelrmse(last_arr):.3f}")
    print(f"  oracle-level       pooled {pooled(tmean):.3f}  levelRMSE 0", flush=True)

    for R in (25, 35, 45):
        for temp in (0.5, 1.0, 2.0):
            lv = [level_grmean(P, R=R, temp=temp) for P in Ps]
            print(f"  grmean R={R} T={temp:>3}  pooled {pooled(lv):.3f}  levelRMSE {levelrmse(lv):.3f}", flush=True)

    for R in (25, 35):
        for band in (6, 10, 15):
            lv = [level_quant(P, R=R, band=band, temp=1.0) for P in Ps]
            print(f"  quant  R={R} band={band:>2} pooled {pooled(lv):.3f}  levelRMSE {levelrmse(lv):.3f}", flush=True)

    # best grmean, shrink toward last (regularize the level move)
    lv = np.array([level_grmean(P, R=35, temp=1.0) for P in Ps])
    for sh in (0.3, 0.5, 0.7):
        blended = sh * lv + (1 - sh) * last_arr
        print(f"  grmean*{sh}+ff       pooled {pooled(blended):.3f}  levelRMSE {levelrmse(blended):.3f}", flush=True)


if __name__ == "__main__":
    main()
