"""TREND/SLOPE probe — is the per-well eval-zone drift rate recoverable from test-available
features? Oracle says know-the-slope = 6.70 vs constant 9.04, so a real slope signal could be
worth ~1 RMSE point. Mirrors tools/level_probe.py.

Target: true_slope = OLS slope of eval TVT vs (MD - MD_ps), in ft-TVT per 1000 ft MD.
Candidate test-available per-well features:
  v50, v200            pre-PS per-row TVT velocity (does the bit keep drifting at that rate?)
  ps_tvt_slope         OLS slope of last 150 known TVT vs MD (pre-PS drift, MD-scaled)
  eval_z_slope         OLS slope of eval Z vs MD (we KNOW eval Z; TVT~Z minus bed dip)
  known_z_slope        OLS slope of last 150 known Z vs MD
  gr_off, eval_len, known_range

Reports OOF R^2 (5-fold) for velocity-only / +Z-trajectory / all, and raw corr per feature.

    uv run python -u tools/slope_probe.py
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from src.data import load_split, TARGET_INPUT
from src.model import extract_level_features, LEVEL_FEATURES, _gr_offset_raw

t0 = time.time()
def log(m): print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


def ols_slope(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return 0.0
    x, y = x[m], y[m]; xc = x - x.mean()
    d = float(np.sum(xc * xc))
    return float(np.sum(xc * (y - y.mean())) / d) if d > 1e-9 else 0.0


def build():
    wells = load_split(Path("data/raw/train"))
    rows = []
    for w in wells:
        h = w.horizontal
        em = w.eval_mask.to_numpy()
        kn = h[h[TARGET_INPUT].notna()]
        if em.sum() < 10 or len(kn) < 200 or "TVT" not in h.columns:
            continue
        md = h["MD"].to_numpy(float); z = h["Z"].to_numpy(float); tv = h["TVT"].to_numpy(float)
        md_ps = float(kn["MD"].iloc[-1])
        # target: eval TVT slope per 1000 ft MD
        true_slope = ols_slope(md[em] - md_ps, tv[em]) * 1000.0
        # pre-PS drifts (MD-scaled), last 150 known rows
        k = kn.index.values[-150:]
        ps_tvt_slope = ols_slope(md[k], tv[k]) * 1000.0
        known_z_slope = ols_slope(md[k], z[k]) * 1000.0
        eval_z_slope = ols_slope(md[em] - md_ps, z[em]) * 1000.0
        f = extract_level_features(w)
        rows.append(dict(slope=true_slope, ps_tvt_slope=ps_tvt_slope,
                         known_z_slope=known_z_slope, eval_z_slope=eval_z_slope,
                         v50=float(f["v50"]), v200=float(f["v200"]),
                         gr_off=float(f["gr_off"]), eval_len=float(f["eval_len"]),
                         known_range=float(f["known_range"])))
    return rows


def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def oof(X, y):
    p = np.zeros(len(y))
    for tr, va in KFold(5, shuffle=True, random_state=0).split(X):
        p[va] = Ridge(1.0).fit(X[tr], y[tr]).predict(X[va])
    return p


def main():
    log("building slope table...")
    rows = build()
    y = np.array([r["slope"] for r in rows])
    log(f"{len(rows)} wells; slope mean={y.mean():.3f} std={y.std():.3f} (TVT ft / 1000ft MD)")

    feats = ["v50", "v200", "ps_tvt_slope", "known_z_slope", "eval_z_slope",
             "gr_off", "eval_len", "known_range"]
    M = {k: np.array([r[k] for r in rows]) for k in feats}
    print("\n=== raw corr with true eval slope ===", flush=True)
    for k in feats:
        print(f"  {k:16s} corr={np.corrcoef(M[k], y)[0,1]:+.3f}", flush=True)

    groups = {
        "velocity [v50,v200]": ["v50", "v200"],
        "+ pre-PS TVT slope":  ["v50", "v200", "ps_tvt_slope"],
        "+ Z-trajectory":      ["v50", "v200", "ps_tvt_slope", "known_z_slope", "eval_z_slope"],
        "all 8 feats":         feats,
    }
    print("\n=== per-well SLOPE OOF R^2 (5-fold) ===", flush=True)
    for name, ks in groups.items():
        X = np.column_stack([M[k] for k in ks])
        print(f"  {name:24s} R2={r2(y, oof(X, y)):+.3f}", flush=True)


if __name__ == "__main__":
    main()
