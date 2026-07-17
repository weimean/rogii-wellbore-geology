"""NEW window/sequence GR-alignment features — genuinely different from the point-wise multi-offset
grids already in src.model._rich_cols.

BACKGROUND (measured here): TVT is ~11-12k (absolute stratigraphic depth) and over the eval zone Z
varies 100-240 ft while TVT varies only 15-50 ft — the beds dip nearly parallel to the wellbore, so
TVT = datum - Z is FALSE (per-well corr(U0-Z, TVT) swings 0.93..-0.11). Confirmed: a constant-U datum
scan gives pooled RMSE ~87-104 (see tools/gr_datum_probe.py). So any NCC/DTW that infers TVT from Z is
DEAD, and a monotone-DTW warp to a TVT-sorted reference is invalid because TVT(MD) is non-monotone
(that latent-wander model is exactly the pipeline's HMM).

The one Z-free, non-monotone-compatible NEW family: WINDOWED-NCC REFINEMENT OF THE HMM PATH. For each
eval row, slide a box window of horizontal GR against the reference GR sampled at (hmm_tvt + delta)
over a small delta grid; emit the argmax offset (a windowed correction to the HMM), the peak windowed
NCC (match confidence) and its sharpness. This is a windowed/shape summary (the existing grids are
per-row point residuals) and rides the HMM's non-monotone path instead of Z.

    uv run python -u tools/gr_align_ncc.py sanity 150   # refined-vs-HMM-vs-ffill pooled RMSE
    uv run python -u tools/gr_align_ncc.py build        # full 773-well cache -> /tmp/rogii_ncc.npz
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from src.data import load_split
from src.model import _stack_features, hmm_posterior

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)
TI = "TVT_input"

WINDOWS = (10, 25, 50)          # box radii (rows)
NDELTA = 25; DRANGE = 12.0      # TVT-offset grid around the HMM path: +-12 ft
EPS = 1e-6
REFS = ("tw", "self")
FEAT_KINDS = ("off", "pk", "sh")
FEAT_NAMES = [f"ncc_{r}_{k}_{w}" for r in REFS for w in WINDOWS for k in FEAT_KINDS]


def _prep(well):
    """eval GR + HMM baseline (Z-free, non-monotone) + typewell/self GR(TVT) references."""
    h, t = well.horizontal, well.typewell
    em = well.eval_mask.to_numpy()
    if em.sum() < 20 or "TVT" not in h.columns:
        return None
    kn = ~em
    gr = h["GR"].interpolate(limit_direction="both").to_numpy(float)
    kt_full = h[TI].to_numpy(float)
    kt = kt_full[kn]; kg = gr[kn]
    ms = np.isfinite(kt) & np.isfinite(kg)
    if ms.sum() < 30:
        return None
    # self reference: GR(TVT) over the known zone (no gain cal — same tool)
    o = np.argsort(kt[ms]); st = kt[ms][o]; sg = kg[ms][o]
    st, ui = np.unique(st, return_index=True); sg = gaussian_filter1d(sg[ui], 2)
    if len(st) < 10:
        return None
    # typewell reference, affine heel-cal to the horizontal scale
    tw_tvt = t["TVT"].to_numpy(float); tw_gr = t["GR"].ffill().bfill().to_numpy(float)
    ot = np.argsort(tw_tvt); tw_tvt = tw_tvt[ot]; tw_gr = gaussian_filter1d(tw_gr[ot], 3)
    a, b = np.polyfit(np.interp(kt[ms], tw_tvt, tw_gr), kg[ms], 1)
    if not np.isfinite(a) or abs(a) < 1e-6:
        a, b = 1.0, 0.0
    R_tw = a * tw_gr + b
    hp, _ = hmm_posterior(well)
    ei = np.where(em)[0]
    ge = gaussian_filter1d(gr[ei], 3)
    return dict(ge=ge, hp=hp[em].astype(float), n=int(em.sum()),
                self_t=st, R_self=sg, tw_t=tw_tvt, R_tw=R_tw)


def ncc_refine(ge, base, ref_t, ref_g, w):
    """Per-row windowed-NCC refinement of `base` (HMM TVT). For each delta, pred_j = ref_g(base_j+delta);
    box-window NCC(ge, pred) radius w. Returns (off*, peak, sharp) per row."""
    n = len(ge); size = 2 * w + 1
    ds = np.linspace(-DRANGE, DRANGE, NDELTA)
    mg = uniform_filter1d(ge, size, mode="nearest")
    vg = np.maximum(uniform_filter1d(ge * ge, size, mode="nearest") - mg * mg, 0.0)
    ncc = np.empty((NDELTA, n), np.float32)
    for k in range(NDELTA):
        pred = np.interp(base + ds[k], ref_t, ref_g)
        mp = uniform_filter1d(pred, size, mode="nearest")
        vp = np.maximum(uniform_filter1d(pred * pred, size, mode="nearest") - mp * mp, 0.0)
        mgp = uniform_filter1d(ge * pred, size, mode="nearest")
        ncc[k] = (mgp - mg * mp) / np.sqrt(vg * vp + EPS)
    ki = np.argmax(ncc, 0)
    peak = ncc[ki, np.arange(n)]
    off = ds[ki]
    sharp = peak - ncc.mean(0)
    return off.astype(np.float32), peak.astype(np.float32), sharp.astype(np.float32)


def build_well(well):
    P = _prep(well)
    if P is None:
        return None
    out = {}
    for r, (rt, rg) in (("tw", (P["tw_t"], P["R_tw"])), ("self", (P["self_t"], P["R_self"]))):
        for w in WINDOWS:
            off, pk, sh = ncc_refine(P["ge"], P["hp"], rt, rg, w)
            out[f"ncc_{r}_off_{w}"] = off; out[f"ncc_{r}_pk_{w}"] = pk; out[f"ncc_{r}_sh_{w}"] = sh
    out["_hp"] = P["hp"].astype(np.float32)
    return out


def sanity(nw):
    wells = load_split(Path("data/raw/train"), limit=nw)
    truth = []; ff = []; hpv = []; refined = {w: [] for w in WINDOWS}; refined_s = {w: [] for w in WINDOWS}
    for w in wells:
        h = w.horizontal; em = w.eval_mask.to_numpy()
        if em.sum() < 20 or "TVT" not in h.columns:
            continue
        o = build_well(w)
        if o is None:
            continue
        truth.append(h["TVT"].to_numpy()[em]); hp = o["_hp"]
        ff.append(np.full(em.sum(), float(h[h[TI].notna()][TI].iloc[-1]))); hpv.append(hp)
        for ww in WINDOWS:
            refined[ww].append(hp + o[f"ncc_tw_off_{ww}"])
            refined_s[ww].append(hp + o[f"ncc_self_off_{ww}"])
    truth = np.concatenate(truth)
    def rmse(p): return float(np.sqrt(np.mean((truth - p) ** 2)))
    log(f"{len(truth)} eval rows")
    print(f"  ffill:            {rmse(np.concatenate(ff)):.3f}", flush=True)
    print(f"  HMM baseline:     {rmse(np.concatenate(hpv)):.3f}", flush=True)
    for ww in WINDOWS:
        print(f"  HMM+tw_off  w={ww:<3} {rmse(np.concatenate(refined[ww])):.3f}"
              f"   HMM+self_off {rmse(np.concatenate(refined_s[ww])):.3f}", flush=True)


def build():
    wells = load_split(Path("data/raw/train"))
    ref = np.load("/tmp/rogii_feats.npz")["truth"]
    acc = {n: [] for n in FEAT_NAMES}
    kept = 0
    for k, w in enumerate(wells):
        em = w.eval_mask.to_numpy()
        if em.sum() == 0:
            continue
        n = int(em.sum())
        f = _stack_features(w, None, anchor_override=np.zeros(n))
        if f is None or "truth" not in f:
            continue
        o = build_well(w)
        for nm in FEAT_NAMES:
            acc[nm].append(o[nm] if o is not None else np.zeros(n, np.float32))
        kept += 1
        if (k + 1) % 100 == 0:
            log(f"  {k+1}/{len(wells)} ({kept} kept)")
    acc = {nm: np.concatenate(v) for nm, v in acc.items()}
    assert len(acc[FEAT_NAMES[0]]) == len(ref), f"align mismatch {len(acc[FEAT_NAMES[0]])} vs {len(ref)}"
    np.savez_compressed("/tmp/rogii_ncc.npz", **acc)
    log(f"saved {len(acc[FEAT_NAMES[0]])} rows, {len(FEAT_NAMES)} feats -> /tmp/rogii_ncc.npz")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "sanity"
    if mode == "sanity":
        sanity(int(sys.argv[2]) if len(sys.argv) > 2 else 150)
    else:
        build()
