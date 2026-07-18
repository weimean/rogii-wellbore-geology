"""Constrained beam-search DP dip-correlation (notes/22, the top-team 'auto-dip' method). A Viterbi/
DTW over a TVT-vs-MD trellis: states = typewell TVT-grid indices, stages = smoothed horizontal-GR
samples along MD. Cost = (gr_h - tw_gr(TVT))^2/emit + move*|dTVT|, dip bounded to +-max_d cells/stage,
beam-pruned. Run at several stiffness settings -> a FAMILY of TVT paths. This is the constrained cousin
of our per-point NCC-peak (which was noise) and of our smooth 2nd-order HMM (one tuned path).

GO/NO-GO: is a single tuned beam path a SENSIBLE predictor (beats ffill, ~ HMM)? If garbage -> stop.

    uv run python -u tools/beam_dp_probe.py sanity 250
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from src.data import load_split
from src.model import hmm_posterior

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)
TI = "TVT_input"
STEP = 0.5; BAND = 90.0

# stiffness regimes: (beam_size, move_cost, emit_scale, gr_smooth, max_d)
CONFIGS = {
    "cons":   dict(beam=10, move=20.0, emit=144.0, sm=2.0, md=2),
    "loose":  dict(beam=12, move=8.0,  emit=64.0,  sm=2.0, md=2),
    "vcons":  dict(beam=8,  move=35.0, emit=220.0, sm=1.0, md=2),
    "sm5":    dict(beam=10, move=14.0, emit=90.0,  sm=5.0, md=2),
    "vloose": dict(beam=20, move=4.0,  emit=36.0,  sm=3.0, md=3),
    "mid":    dict(beam=12, move=12.0, emit=100.0, sm=3.0, md=2),
    "stiff":  dict(beam=15, move=25.0, emit=180.0, sm=2.0, md=2),
}


def beam_path(gr_b, gr_grid, start_idx, cfg):
    """Beam-search DP. gr_b:(S,) smoothed horizontal GR per stage; gr_grid:(P,) typewell GR per TVT
    cell. Returns path grid-indices (S,)."""
    S = len(gr_b); P = len(gr_grid); B = cfg["beam"]; md = cfg["md"]
    move = cfg["move"]; emit = cfg["emit"]
    ds = np.arange(-md, md + 1)
    emit_all = (gr_b[:, None] - gr_grid[None, :]) ** 2 / emit          # (S,P)
    # beam state
    bs = np.array([start_idx]); bc = np.array([emit_all[0, start_idx]])
    bp = [np.array([-1])]; bstates = [bs.copy()]
    for s in range(1, S):
        cand_i = (bs[:, None] + ds[None, :]).ravel()                   # (B*nd,)
        parent = np.repeat(np.arange(len(bs)), len(ds))
        cost = (bc[:, None] + move * np.abs(ds)[None, :]).ravel()
        ok = (cand_i >= 0) & (cand_i < P)
        cand_i = cand_i[ok]; parent = parent[ok]; cost = cost[ok]
        cost = cost + emit_all[s, cand_i]
        # best-per-state
        order = np.argsort(cost)
        cand_i, parent, cost = cand_i[order], parent[order], cost[order]
        _, uniq = np.unique(cand_i, return_index=True)
        cand_i, parent, cost = cand_i[uniq], parent[uniq], cost[uniq]
        # prune to beam
        if len(cost) > B:
            keep = np.argsort(cost)[:B]
            cand_i, parent, cost = cand_i[keep], parent[keep], cost[keep]
        bs, bc = cand_i, cost; bstates.append(bs.copy()); bp.append(parent)
    # backtrack from global best
    j = int(np.argmin(bc)); path = np.empty(S, int)
    for s in range(S - 1, -1, -1):
        path[s] = bstates[s][j]; j = bp[s][j]
        if j < 0 and s > 0:
            j = 0
    return path


def run_well(well, cfg_names):
    h, t = well.horizontal, well.typewell
    kn = h[h[TI].notna()]; ev = h[h[TI].isna()]
    if len(ev) < 20 or len(kn) < 200 or "TVT" not in h.columns:
        return None
    tw_tvt = t["TVT"].to_numpy(float); tw_gr = t["GR"].ffill().bfill().to_numpy(float)
    ot = np.argsort(tw_tvt); tw_tvt = tw_tvt[ot]; tw_gr = tw_gr[ot]
    last = float(kn[TI].iloc[-1])
    gmin = max(tw_tvt.min() - 40, last - BAND); gmax = min(tw_tvt.max() + 40, last + BAND)
    grid = np.arange(gmin, gmax + STEP, STEP); gr_grid = np.interp(grid, tw_tvt, tw_gr)
    start_idx = int(np.clip(round((last - gmin) / STEP), 0, len(grid) - 1))
    ei = ev.index.values
    gr_all = h["GR"].interpolate(limit_direction="both").fillna(np.nanmean(tw_gr)).to_numpy(float)
    gr_e = gr_all[ei]; n = len(ei)
    S = min(300, n); edges = np.linspace(0, n, S + 1).astype(int)
    out = {"truth": h["TVT"].to_numpy(float)[ei], "last": last, "n": n}
    for name in cfg_names:
        cfg = CONFIGS[name]
        gr_s = gaussian_filter1d(gr_e, cfg["sm"])
        gr_b = np.array([gr_s[a:b].mean() for a, b in zip(edges[:-1], edges[1:])])
        # affine-cal known GR to typewell scale for a fair emission
        path = beam_path(gr_b, gr_grid, start_idx, cfg)
        tvt_b = grid[path]
        centers = 0.5 * (edges[:-1] + edges[1:])
        out[name] = np.interp(np.arange(n), centers, tvt_b)
    return out


def sanity(nw):
    wells = load_split(Path("data/raw/train"), limit=nw)
    names = list(CONFIGS)
    aT = []; aFF = []; aHMM = []; A = {nm: [] for nm in names}
    kept = 0
    for w in wells:
        r = run_well(w, names)
        if r is None:
            continue
        hp, _ = hmm_posterior(w); em = w.eval_mask.to_numpy()
        aT.append(r["truth"]); aFF.append(np.full(r["n"], r["last"])); aHMM.append(hp[em])
        for nm in names:
            A[nm].append(r[nm])
        kept += 1
    aT = np.concatenate(aT); aFF = np.concatenate(aFF); aHMM = np.concatenate(aHMM)
    def rmse(p): return float(np.sqrt(np.mean((aT - p) ** 2)))
    log(f"{len(aT)} eval rows, {kept} wells")
    print(f"  ffill:        {rmse(aFF):.3f}", flush=True)
    print(f"  HMM (ours):   {rmse(aHMM):.3f}", flush=True)
    for nm in names:
        print(f"  beam[{nm:6s}]: {rmse(np.concatenate(A[nm])):.3f}", flush=True)
    # simple mean-of-family
    fam = np.mean([np.concatenate(A[nm]) for nm in names], 0)
    print(f"  beam mean-of-{len(names)}: {rmse(fam):.3f}", flush=True)


def build():
    """Full 773-well beam-family cache aligned to /tmp/rogii_feats.npz -> /tmp/rogii_beam.npz.
    Features per eval row: 7 raw beam TVT paths + family mean/std (disagreement) + GR-misfit."""
    from src.model import _stack_features
    names = list(CONFIGS)
    wells = load_split(Path("data/raw/train"))
    ref = np.load("/tmp/rogii_feats.npz")["truth"]
    FEATS = [f"beam_{nm}" for nm in names] + ["beam_mean", "beam_std", "beam_misfit"]
    acc = {f: [] for f in FEATS}; kept = 0
    for k, w in enumerate(wells):
        em = w.eval_mask.to_numpy()
        if em.sum() == 0:
            continue
        n = int(em.sum())
        f = _stack_features(w, None, anchor_override=np.zeros(n))
        if f is None or "truth" not in f:
            continue
        r = run_well(w, names)
        if r is None:
            for f_ in FEATS:
                acc[f_].append(np.zeros(n, np.float32))
        else:
            paths = np.stack([r[nm] for nm in names], 0)              # (7, n)
            for nm in names:
                acc[f"beam_{nm}"].append(r[nm].astype(np.float32))
            acc["beam_mean"].append(paths.mean(0).astype(np.float32))
            acc["beam_std"].append(paths.std(0).astype(np.float32))
            # GR-misfit of the family-mean path vs typewell GR
            h, t = w.horizontal, w.typewell
            tw_tvt = t["TVT"].to_numpy(float); tw_gr = t["GR"].ffill().bfill().to_numpy(float)
            ot = np.argsort(tw_tvt); tw_tvt = tw_tvt[ot]; tw_gr = tw_gr[ot]
            ei = np.where(em)[0]
            gr_e = h["GR"].interpolate(limit_direction="both").fillna(np.nanmean(tw_gr)).to_numpy(float)[ei]
            mf = np.abs(gr_e - np.interp(paths.mean(0), tw_tvt, tw_gr)).astype(np.float32)
            acc["beam_misfit"].append(mf)
        kept += 1
        if (k + 1) % 100 == 0:
            log(f"  {k+1}/{len(wells)} ({kept} kept)")
    acc = {f: np.concatenate(v) for f, v in acc.items()}
    assert len(acc[FEATS[0]]) == len(ref), f"align mismatch {len(acc[FEATS[0]])} vs {len(ref)}"
    np.savez_compressed("/tmp/rogii_beam.npz", **acc)
    log(f"saved {len(acc[FEATS[0]])} rows, {len(FEATS)} feats -> /tmp/rogii_beam.npz")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "sanity"
    if mode == "build":
        build()
    else:
        sanity(int(sys.argv[2]) if len(sys.argv) > 2 else 250)
