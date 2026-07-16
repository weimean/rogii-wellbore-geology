"""Two questions:
 (A) Does a full GR-SEQUENCE datum scan pin the per-well LEVEL better than mean/quantile matching?
     For each candidate level c, predict a per-row TVT = c + wiggle_i where wiggle_i is a smoothed
     local-linearization inversion of GR around c; score the whole predicted GR sequence vs observed.
 (B) Does adding local-linearization WIGGLE on top of the best level beat the flat level?

ffill=15.65, oracle-level=8.65, best-level-so-far ~14.46 (150w).

    uv run python -u tools/gr_seq_probe.py [NWELLS]
"""
from __future__ import annotations
import os, sys
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
    # smooth the eval GR (denoise) for wiggle inversion
    gn_s = gaussian_filter1d(gn, 8)
    resid = kgr[mm] - (a * tw_at_kn[mm] + b)
    sigma = float(np.clip(np.std(resid), 3.0, 40.0))
    return dict(tw_tvt=tw_tvt, tw_gr=tw_grs, gn=gn, gn_s=gn_s, last=last,
                truth=tvt[em], truemean=float(np.mean(tvt[em])), sigma=sigma, n=int(em.sum()))


def tw_slope(P, c, dz=4.0):
    g1 = np.interp(c + dz, P["tw_tvt"], P["tw_gr"])
    g0 = np.interp(c - dz, P["tw_tvt"], P["tw_gr"])
    return (g1 - g0) / (2 * dz)


def wiggle_at(P, c, gain=1.0, clip=12.0):
    """local-linearization: dTVT = (gn_s - tw_gr(c)) / slope(c), smoothed & clipped."""
    s = tw_slope(P, c)
    if abs(s) < 1e-3:
        return np.zeros(P["n"])
    dg = P["gn_s"] - np.interp(c, P["tw_tvt"], P["tw_gr"])
    dt = gain * dg / s
    dt = np.clip(dt, -clip, clip)
    return gaussian_filter1d(dt, 15)


def seq_scan(P, R=35.0, npts=71, temp=1.0, with_wiggle=True):
    """scan level c; predicted GR sequence = tw_gr(c + wiggle_i); cost vs observed gn."""
    cs = np.linspace(P["last"] - R, P["last"] + R, npts)
    cost = np.empty(len(cs))
    for i, c in enumerate(cs):
        wig = wiggle_at(P, c) if with_wiggle else 0.0
        pred = np.interp(c + wig, P["tw_tvt"], P["tw_gr"])
        cost[i] = np.mean((P["gn"] - pred) ** 2)
    cost /= (2 * P["sigma"] ** 2)
    w = np.exp(-(cost - cost.min()) / temp); w /= w.sum()
    c_hat = float((w * cs).sum())
    wig = wiggle_at(P, c_hat) if with_wiggle else np.zeros(P["n"])
    return c_hat, c_hat + wig


def level_grmean(P, R=35.0, temp=1.0):
    cs = np.linspace(P["last"] - R, P["last"] + R, 141)
    pg = np.interp(cs, P["tw_tvt"], P["tw_gr"])
    cost = (pg - np.nanmean(P["gn"])) ** 2 / (2 * P["sigma"] ** 2)
    w = np.exp(-(cost - cost.min()) / temp); w /= w.sum()
    return float((w * cs).sum())


def main():
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    wells = load_split(Path("data/raw/train"), limit=nw)
    Ps = [P for P in (prep(w) for w in wells) if P is not None]
    truth = np.concatenate([P["truth"] for P in Ps])
    last = np.array([P["last"] for P in Ps])
    tmean = np.array([P["truemean"] for P in Ps])

    def pooled(preds): return np.sqrt(np.mean((truth - np.concatenate(preds)) ** 2))
    def lrmse(lv): return np.sqrt(np.mean((np.array(lv) - tmean) ** 2))

    print(f"=== {len(Ps)} wells, {len(truth)} eval rows ===", flush=True)
    print(f"  ffill        pooled {np.sqrt(np.mean((truth-np.concatenate([np.full(P['n'],P['last']) for P in Ps]))**2)):.3f}")

    # best flat level (grmean)
    gl = [level_grmean(P) for P in Ps]
    print(f"  grmean flat  pooled {pooled([np.full(P['n'],lv) for P,lv in zip(Ps,gl)]):.3f}  levelRMSE {lrmse(gl):.3f}")

    # grmean level + local-lin wiggle on top
    predw = []
    for P, lv in zip(Ps, gl):
        predw.append(lv + wiggle_at(P, lv))
    print(f"  grmean+wiggle pooled {pooled(predw):.3f}", flush=True)

    # sequence datum scan (level via full sequence match, with & without wiggle)
    for ww in (False, True):
        clv = []; preds = []
        for P in Ps:
            c, path = seq_scan(P, with_wiggle=ww)
            clv.append(c); preds.append(path)
        print(f"  seqscan wig={ww}  pooled {pooled(preds):.3f}  levelRMSE {lrmse(clv):.3f}", flush=True)


if __name__ == "__main__":
    main()
