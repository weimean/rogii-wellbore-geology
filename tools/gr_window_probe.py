"""GR-matching formulations for eval-zone TVT. Compare against ffill (15.91 full / ~15.65 on 150).

Decomposition (measured): ffill err 15.65 = LEVEL(RMS 12.9, per-well offset) + WIGGLE(std 7.5).
Physics confirmed: corr(horiz GR, typewell_GR(true TVT)) = 0.65 eval / 0.82 known.

Formulations tested here:
  - level_grmean : pick per-well constant TVT near last_known whose typewell GR == mean eval GR (bimodal hedge)
  - level_ncc    : NCC-register the horizontal GR *pattern* against the typewell to pin absolute TVT level
  - hmm_smooth   : Viterbi over TVT grid, affine-calibrated GR emission + smoothness (captures level+wiggle)
  - blends with ffill

    uv run python -u tools/gr_window_probe.py [NWELLS]
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
    ktvt = tvt[kn]; kgr = gr[kn]
    last = ktvt[-1]
    # affine calibrate horizontal GR -> typewell scale using KNOWN part
    tw_at_kn = np.interp(ktvt, tw_tvt, tw_grs)
    mm = np.isfinite(tw_at_kn) & np.isfinite(kgr)
    if mm.sum() < 30:
        return None
    a, b = np.polyfit(tw_at_kn[mm], kgr[mm], 1)
    if not np.isfinite(a) or abs(a) < 1e-6:
        a, b = 1.0, 0.0
    gr_e = gr[em]
    gn = (gr_e - b) / a  # eval horiz GR on typewell scale
    resid = kgr[mm] - (a * tw_at_kn[mm] + b)
    sigma = float(np.clip(np.std(resid), 3.0, 40.0))
    return dict(tw_tvt=tw_tvt, tw_gr=tw_grs, gn=gn, last=last,
                truth=tvt[em], em=em, sigma=sigma, n=int(em.sum()))


def level_grmean(P, R=35.0, temp=1.0):
    """constant TVT near last_known whose typewell GR matches mean eval GR; bimodal softmax hedge."""
    tw_tvt, tw_gr = P["tw_tvt"], P["tw_gr"]
    last = P["last"]
    cs = np.linspace(last - R, last + R, 141)
    pg = np.interp(cs, tw_tvt, tw_gr)
    target = np.nanmean(P["gn"])
    cost = (pg - target) ** 2 / (2 * P["sigma"] ** 2)
    w = np.exp(-(cost - cost.min()) / temp); w /= w.sum()
    c = float((w * cs).sum())
    return np.full(P["n"], c)


def hmm_smooth(P, R=45.0, step=1.5, tau=2.5, huber=3.0, blend_ff=0.0):
    """Viterbi over TVT grid around last_known. Emission = affine-cal GR misfit; transition = smoothness."""
    tw_tvt, tw_gr = P["tw_tvt"], P["tw_gr"]
    last = P["last"]; gn = P["gn"]; sigma = P["sigma"]; n = P["n"]
    grid = np.arange(last - R, last + R + 1e-6, step)
    S = len(grid)
    pred_gr = np.interp(grid, tw_tvt, tw_gr)  # (S,)
    # emission cost (n, S)
    r = np.clip((gn[:, None] - pred_gr[None, :]) / sigma, -huber, huber)
    E = r ** 2
    # transition penalty on state index change (smoothness): quadratic on delta TVT
    dgrid = (grid[None, :] - grid[:, None]) / tau
    T = 0.5 * dgrid ** 2  # (S_from, S_to)
    # prior pulling first state toward last (known continuity)
    E[0] += 0.5 * ((grid - last) / 8.0) ** 2
    # forward DP
    V = E[0].copy()
    back = np.zeros((n, S), dtype=np.int32)
    for i in range(1, n):
        # cost to arrive at each state = min_prev V_prev + T[prev,to]
        tot = V[:, None] + T  # (S_from, S_to)
        j = np.argmin(tot, axis=0)
        V = tot[j, np.arange(S)] + E[i]
        back[i] = j
    path = np.zeros(n, dtype=np.int32)
    path[-1] = int(np.argmin(V))
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    tvt = grid[path]
    if blend_ff > 0:
        tvt = (1 - blend_ff) * tvt + blend_ff * last
    return tvt


def main():
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    wells = load_split(Path("data/raw/train"), limit=nw)
    variants = {"ffill": [], "level_grmean": [], "hmm_smooth": [],
                "hmm_blend50": [], "level_blend": []}
    truth = []
    nused = 0
    for w in wells:
        P = prep(w)
        if P is None:
            continue
        nused += 1
        truth.append(P["truth"])
        last = P["last"]; n = P["n"]
        variants["ffill"].append(np.full(n, last))
        lg = level_grmean(P)
        variants["level_grmean"].append(lg)
        variants["level_blend"].append(0.5 * lg + 0.5 * last)
        hs = hmm_smooth(P)
        variants["hmm_smooth"].append(hs)
        variants["hmm_blend50"].append(0.5 * hs + 0.5 * last)
    truth = np.concatenate(truth)
    def rmse(k): return np.sqrt(np.mean((truth - np.concatenate(variants[k])) ** 2))
    print(f"\n=== pooled over {len(truth)} eval rows, {nused} wells ===", flush=True)
    for k in variants:
        print(f"  {k:16s} {rmse(k):.3f}", flush=True)


if __name__ == "__main__":
    main()
