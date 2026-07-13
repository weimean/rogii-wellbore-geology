"""Family 1 — global typewell LEVEL alignment (the make-or-break probe).

The eval-zone GR (as a function of MD) is a shifted, affine-scaled copy of the typewell GR
(as a function of absolute TVT). Fit per well a (level L, slope s) so the predicted typewell
TVT along the eval path,  tvt_hat = L + s*(MD - MD_ps),  makes the typewell GR sampled along
tvt_hat best CORRELATE (Pearson, so calibration-invariant) with the eval GR. The implied
eval-mean level pins the per-well offset directly from a GLOBAL sequence fit — unlike the
per-point HMM whose mean is actually worse than ffill (offset R2 = -0.078).

Reports OOF offset R^2 (5-fold, 773 wells):
  anchor        : current 5-feat Ridge (baseline 0.098)
  align (raw)   : predicted offset from the global fit, standalone
  anchor+align  : Ridge on the 5 feats + align features
Decision: family 1 is real iff anchor+align clears ~0.30+ (vs 0.098) — that would propagate
into the stack (which currently reaches 0.337 with the whole HMM+PF+GBM apparatus).

    uv run python -u tools/level_align.py
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from src.data import load_split, TARGET_INPUT
from src.model import LEVEL_FEATURES, extract_level_features

t0 = time.time()
def log(m): print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)

MAXPTS = 400        # subsample the eval sequence for the grid search
L_HALF, L_STEP = 45.0, 1.5      # level search: last +/- 45 ft
DRIFT, S_STEPS = 40.0, 11       # slope search: total eval drift within +/- 40 ft


def align_well(w, center_off=0.0, l_half=L_HALF, l_step=L_STEP):
    """Return dict of family-1 features for one well, or None.
    Search level in [last+center_off-l_half, last+center_off+l_half] — pass a narrow band
    around an anchor prediction (center_off) to suppress quasi-periodic GR aliasing."""
    h = w.horizontal
    kn = h[h[TARGET_INPUT].notna()]
    em = w.eval_mask.to_numpy()
    if em.sum() == 0 or len(kn) < 200:
        return None
    last = float(kn[TARGET_INPUT].values[-1])
    md_ps = float(kn["MD"].iloc[-1])
    t = w.typewell
    tw_tvt = t["TVT"].to_numpy(float); tw_gr = t["GR"].to_numpy(float)
    ok = np.isfinite(tw_tvt) & np.isfinite(tw_gr)
    tw_tvt, tw_gr = tw_tvt[ok], tw_gr[ok]
    order = np.argsort(tw_tvt); tw_tvt, tw_gr = tw_tvt[order], tw_gr[order]
    tw_gr = gaussian_filter1d(tw_gr, 3)

    md = h["MD"].to_numpy(float)[em]
    gr = h["GR"].interpolate(limit_direction="both").to_numpy(float)[em]
    gr = gaussian_filter1d(gr, 3)
    dmd = md - md_ps
    # subsample for speed
    if len(md) > MAXPTS:
        sel = np.linspace(0, len(md) - 1, MAXPTS).astype(int)
        dmd_s, gr_s = dmd[sel], gr[sel]
    else:
        dmd_s, gr_s = dmd, gr
    gn = gr_s - gr_s.mean()
    gn_norm = np.sqrt(np.sum(gn * gn)) + 1e-9

    Dmd = float(np.abs(dmd_s).max()) + 1e-9
    Ls = last + center_off + np.arange(-l_half, l_half + 1e-6, l_step)
    Ss = np.linspace(-DRIFT / Dmd, DRIFT / Dmd, S_STEPS)
    best = (-2.0, last, 0.0)
    for s in Ss:
        base = s * dmd_s
        for L in Ls:
            exp = np.interp(L + base, tw_tvt, tw_gr)
            en = exp - exp.mean()
            denom = gn_norm * (np.sqrt(np.sum(en * en)) + 1e-9)
            c = float(np.sum(gn * en) / denom)
            if c > best[0]:
                best = (c, L, s)
    qual, Lstar, sstar = best[0], best[1], best[2]
    implied_mean = Lstar + sstar * float(dmd.mean())
    return dict(align_off=implied_mean - last, align_qual=qual,
                align_slope=sstar * 1000.0, align_L=Lstar - last)


def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def main():
    wells = load_split(Path("data/raw/train"))
    log(f"loaded {len(wells)} wells; aligning...")
    recs = []
    for i, w in enumerate(wells):
        h = w.horizontal
        if "TVT" not in h.columns:
            continue
        a = align_well(w)
        if a is None:
            continue
        em = w.eval_mask.to_numpy()
        kn = h[h[TARGET_INPUT].notna()]
        last = float(kn[TARGET_INPUT].values[-1])
        offset = float(np.nanmean(h["TVT"].to_numpy(float)[em]) - last)
        feats = extract_level_features(w)
        recs.append(dict(offset=offset, **a, **{k: float(feats[k]) for k in LEVEL_FEATURES}))
        if (i + 1) % 150 == 0:
            log(f"  {i+1}/{len(wells)}")

    y = np.array([r["offset"] for r in recs])
    AL = ["align_off", "align_qual", "align_slope", "align_L"]
    Xa = np.array([[r[k] for k in AL] for r in recs])
    Xb = np.array([[r[k] for k in LEVEL_FEATURES] for r in recs])
    log(f"{len(y)} wells; align_qual median={np.median(Xa[:,1]):.3f}")

    kf = KFold(5, shuffle=True, random_state=0)
    b = np.zeros(len(y)); comb = np.zeros(len(y))
    for tr, va in kf.split(Xb):
        b[va] = Ridge(1.0).fit(Xb[tr], y[tr]).predict(Xb[va])
        Xc = np.column_stack([Xb, Xa])
        comb[va] = Ridge(1.0).fit(Xc[tr], y[tr]).predict(Xc[va])
    raw_off = Xa[:, 0]  # align_off standalone as an offset prediction

    print("\n=== per-well offset OOF R^2 (773 wells, 5-fold) ===", flush=True)
    print(f"  anchor (5 feats):     R2={r2(y, b):+.3f}", flush=True)
    print(f"  align_off (raw):      R2={r2(y, raw_off):+.3f}  (corr={np.corrcoef(raw_off,y)[0,1]:+.3f})", flush=True)
    print(f"  anchor + align:       R2={r2(y, comb):+.3f}", flush=True)

    # --- band-limited, anchor-seeded alignment (suppress aliasing) ---
    log("band-limited anchor-seeded alignment...")
    good = [w for w in wells if "TVT" in w.horizontal.columns and align_well(w) is not None]
    # OOF anchor offset per good well (to seed the band)
    Xb_g = np.array([[float(extract_level_features(w)[k]) for k in LEVEL_FEATURES] for w in good])
    y_g = np.array([float(np.nanmean(w.horizontal["TVT"].to_numpy(float)[w.eval_mask.to_numpy()])
                          - w.horizontal[w.horizontal[TARGET_INPUT].notna()][TARGET_INPUT].values[-1])
                    for w in good])
    seed = np.zeros(len(good))
    for tr, va in kf.split(Xb_g):
        m = Ridge(1.0).fit(Xb_g[tr], y_g[tr])
        for j in va:
            seed[j] = m.predict(Xb_g[j:j+1])[0]
    ref_off = np.array([align_well(good[j], center_off=seed[j], l_half=12.0, l_step=0.75)["align_off"]
                        for j in range(len(good))])
    band = np.zeros(len(good))
    Xr = np.column_stack([Xb_g, ref_off])
    for tr, va in kf.split(Xr):
        band[va] = Ridge(1.0).fit(Xr[tr], y_g[tr]).predict(Xr[va])
    print(f"  refined_off (raw):    R2={r2(y_g, ref_off):+.3f}  (corr={np.corrcoef(ref_off,y_g)[0,1]:+.3f})", flush=True)
    print(f"  anchor + refined:     R2={r2(y_g, band):+.3f}", flush=True)


if __name__ == "__main__":
    main()
