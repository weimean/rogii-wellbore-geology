"""GLOBAL NCC-peak-TVT feature family (notes/20). Genuinely different from the EXISTING
/tmp/rogii_ncc.npz, which only searched a +-12 ft local REFINEMENT of the HMM path (-0.016,
redundant). Here the Delta search is GLOBAL (+-DRANGE, wide), the reference is the FULL typewell /
self GR(TVT) profile, and we emit the ABSOLUTE peak-TVT (a direct GR->TVT level estimate that can
correct an HMM level bias), the peak confidence, the 2nd-peak-TVT (bimodality/ambiguity), sharpness,
and cross-window / cross-ref agreement. Multi-window (radii 8/15/30/60) x multi-smoothing (s=1/3/8).

The window's intra-shape comes from the HMM path (base = hmm_tvt path); Delta shifts it globally, so
`pred_j = ref(hmm_tvt_j + Delta)` and windowed-NCC(ge, pred) peaks where the eval GR shape matches the
reference -> peak_tvt = hmm_tvt + Delta*. Z-free, non-monotone-compatible (rides the HMM path).

    uv run python -u tools/gr_peak_probe.py sanity 120   # peak-TVT-as-predictor pooled RMSE
    uv run python -u tools/gr_peak_probe.py build        # full 773-well cache -> /tmp/rogii_grpeak.npz
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

WIN = (8, 15, 30, 60)            # box radii (rows) ~ windows 17/31/61/121
SIGMAS = (1.0, 3.0, 8.0)        # GR smoothings
NDELTA = 61; DRANGE = 90.0      # GLOBAL TVT-offset grid around the HMM path: +-90 ft
EPS = 1e-6
REFS = ("tw", "self")
# per (ref,sigma,window): off1, pk1, off2, pk2, sharp  (5)
PER = ("off1", "pk1", "off2", "pk2", "sharp")
FEAT_NAMES = [f"gp_{r}_{int(s)}_{w}_{k}" for r in REFS for s in SIGMAS for w in WIN for k in PER]
# cross-window / cross-ref agreement (per ref,sigma): std & range of off1 over windows
XNAMES = [f"gpx_{r}_{int(s)}_offstd" for r in REFS for s in SIGMAS] + \
         [f"gpx_{r}_{int(s)}_pkmax" for r in REFS for s in SIGMAS] + \
         ["gpx_refdiff", "gpx_tw_self_off"]
ALLNAMES = FEAT_NAMES + XNAMES


def _prep(well):
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
    # self reference: GR(TVT) over the known zone (no gain cal)
    o = np.argsort(kt[ms]); st = kt[ms][o]; sg0 = kg[ms][o]
    st, ui = np.unique(st, return_index=True); sg0 = sg0[ui]
    if len(st) < 10:
        return None
    # typewell reference, affine heel-cal to the horizontal scale
    tw_tvt = t["TVT"].to_numpy(float); tw_gr0 = t["GR"].ffill().bfill().to_numpy(float)
    ot = np.argsort(tw_tvt); tw_tvt = tw_tvt[ot]; tw_gr0 = tw_gr0[ot]
    a, b = np.polyfit(np.interp(kt[ms], tw_tvt, gaussian_filter1d(tw_gr0, 3)), kg[ms], 1)
    if not np.isfinite(a) or abs(a) < 1e-6:
        a, b = 1.0, 0.0
    hp, _ = hmm_posterior(well)
    ei = np.where(em)[0]
    return dict(gr=gr, ei=ei, hp=hp[em].astype(float), n=int(em.sum()), last=float(kt[-1]),
                self_t=st, self_g0=sg0, tw_t=tw_tvt, tw_g0=tw_gr0, a=a, b=b)


def _ncc_grid(ge, base, ref_t, ref_g, w, ds):
    """windowed-NCC(ge, ref(base+Delta)) for all Delta in ds, box radius w. -> (NDELTA, n)."""
    n = len(ge); size = 2 * w + 1
    mg = uniform_filter1d(ge, size, mode="nearest")
    vg = np.maximum(uniform_filter1d(ge * ge, size, mode="nearest") - mg * mg, 0.0)
    ncc = np.empty((len(ds), n), np.float32)
    for k in range(len(ds)):
        pred = np.interp(base + ds[k], ref_t, ref_g)
        mp = uniform_filter1d(pred, size, mode="nearest")
        vp = np.maximum(uniform_filter1d(pred * pred, size, mode="nearest") - mp * mp, 0.0)
        mgp = uniform_filter1d(ge * pred, size, mode="nearest")
        ncc[k] = (mgp - mg * mp) / np.sqrt(vg * vp + EPS)
    return ncc


def _peaks(ncc, ds):
    """From (NDELTA,n) NCC -> off1, pk1, off2, pk2, sharp (each length n)."""
    n = ncc.shape[1]; ar = np.arange(n)
    ki = np.argmax(ncc, 0); pk1 = ncc[ki, ar]; off1 = ds[ki]
    mean = ncc.mean(0); sharp = pk1 - mean
    # 2nd peak: mask +-3 grid cells around the winner, argmax the rest
    masked = ncc.copy()
    for d in range(-3, 4):
        idx = np.clip(ki + d, 0, ncc.shape[0] - 1)
        masked[idx, ar] = -np.inf
    ki2 = np.argmax(masked, 0); pk2 = ncc[ki2, ar]; off2 = ds[ki2]
    return (off1.astype(np.float32), pk1.astype(np.float32), off2.astype(np.float32),
            pk2.astype(np.float32), sharp.astype(np.float32))


def build_well(well):
    P = _prep(well)
    if P is None:
        return None
    ds = np.linspace(-DRANGE, DRANGE, NDELTA)
    hp = P["hp"]; out = {}
    off1_store = {}   # (ref,sigma,w) -> off1  for cross features
    for r, (rt, rg0, cal) in (("tw", (P["tw_t"], P["tw_g0"], True)),
                              ("self", (P["self_t"], P["self_g0"], False))):
        for s in SIGMAS:
            ge = gaussian_filter1d(P["gr"], s)[P["ei"]]
            rg = gaussian_filter1d(rg0, max(s, 1.0))
            if cal:
                rg = P["a"] * rg + P["b"]
            for w in WIN:
                ncc = _ncc_grid(ge, hp, rt, rg, w, ds)
                o1, p1, o2, p2, sh = _peaks(ncc, ds)
                pre = f"gp_{r}_{int(s)}_{w}_"
                out[pre + "off1"] = o1; out[pre + "pk1"] = p1
                out[pre + "off2"] = o2; out[pre + "pk2"] = p2; out[pre + "sharp"] = sh
                off1_store[(r, s, w)] = o1
    # cross-window agreement + cross-ref
    for r in REFS:
        for s in SIGMAS:
            offs = np.stack([off1_store[(r, s, w)] for w in WIN], 0)
            out[f"gpx_{r}_{int(s)}_offstd"] = offs.std(0).astype(np.float32)
            pks = np.stack([out[f"gp_{r}_{int(s)}_{w}_pk1"] for w in WIN], 0)
            out[f"gpx_{r}_{int(s)}_pkmax"] = pks.max(0).astype(np.float32)
    # tw vs self peak disagreement (sigma 3, w 30 as the representative scale)
    out["gpx_tw_self_off"] = (off1_store[("tw", 3.0, 30)] - off1_store[("self", 3.0, 30)]).astype(np.float32)
    out["gpx_refdiff"] = (out["gp_tw_3_30_pk1"] - out["gp_self_3_30_pk1"]).astype(np.float32)
    out["_hp"] = hp.astype(np.float32); out["_last"] = P["last"]
    return out


def sanity(nw):
    wells = load_split(Path("data/raw/train"), limit=nw)
    truth = []; ff = []; hpv = []
    # peak-TVT as a direct predictor, best/median-of-scales, tw & self
    pk_tw = {w: [] for w in WIN}; pk_self = {w: [] for w in WIN}
    for w in wells:
        h = w.horizontal; em = w.eval_mask.to_numpy()
        if em.sum() < 20 or "TVT" not in h.columns:
            continue
        o = build_well(w)
        if o is None:
            continue
        truth.append(h["TVT"].to_numpy()[em]); hp = o["_hp"]
        ff.append(np.full(em.sum(), o["_last"])); hpv.append(hp)
        for ww in WIN:
            pk_tw[ww].append(hp + o[f"gp_tw_3_{ww}_off1"])
            pk_self[ww].append(hp + o[f"gp_self_3_{ww}_off1"])
    truth = np.concatenate(truth)
    def rmse(p): return float(np.sqrt(np.mean((truth - p) ** 2)))
    log(f"{len(truth)} eval rows, {len(hpv)} wells")
    print(f"  ffill:        {rmse(np.concatenate(ff)):.3f}", flush=True)
    print(f"  HMM baseline: {rmse(np.concatenate(hpv)):.3f}", flush=True)
    for ww in WIN:
        print(f"  peakTVT tw  w={ww:<3} {rmse(np.concatenate(pk_tw[ww])):.3f}"
              f"    peakTVT self {rmse(np.concatenate(pk_self[ww])):.3f}", flush=True)


def build():
    wells = load_split(Path("data/raw/train"))
    ref = np.load("/tmp/rogii_feats.npz")["truth"]
    acc = {n: [] for n in ALLNAMES}
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
        for nm in ALLNAMES:
            acc[nm].append(o[nm] if o is not None else np.zeros(n, np.float32))
        kept += 1
        if (k + 1) % 100 == 0:
            log(f"  {k+1}/{len(wells)} ({kept} kept)")
    acc = {nm: np.concatenate(v) for nm, v in acc.items()}
    assert len(acc[ALLNAMES[0]]) == len(ref), f"align mismatch {len(acc[ALLNAMES[0]])} vs {len(ref)}"
    np.savez_compressed("/tmp/rogii_grpeak.npz", **acc)
    log(f"saved {len(acc[ALLNAMES[0]])} rows, {len(ALLNAMES)} feats -> /tmp/rogii_grpeak.npz")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "sanity"
    if mode == "sanity":
        sanity(int(sys.argv[2]) if len(sys.argv) > 2 else 120)
    else:
        build()
