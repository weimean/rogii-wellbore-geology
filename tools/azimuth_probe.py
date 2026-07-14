"""Domain-motivated probe (from the mycarta toolkit + geosteering theory): does SIGNED DRILLING
AZIMUTH (updip vs downdip) predict the per-well drift the level/slope probes found unpredictable?
Two wells drilled in opposite directions encounter the formation in opposite sequence, so the
azimuth relative to regional dip should set the SIGN of the eval-zone TVT drift.

Our level/slope probes never included azimuth. Test: does azimuth (+ interactions with velocity,
horizontal extent, and absolute X,Y for regional structure) lift offset/slope R^2 above baseline?

    uv run python -u tools/azimuth_probe.py
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from src.data import load_split, TARGET_INPUT
from src.model import extract_level_features, LEVEL_FEATURES

t0 = time.time()
def log(m): print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


def build():
    wells = load_split(Path("data/raw/train"))
    rows = []
    for w in wells:
        h = w.horizontal; em = w.eval_mask.to_numpy(); kn = h[h[TARGET_INPUT].notna()]
        if em.sum() < 10 or len(kn) < 200 or "TVT" not in h.columns:
            continue
        X = h["X"].to_numpy(float); Y = h["Y"].to_numpy(float); MD = h["MD"].to_numpy(float)
        ei = np.where(em)[0]
        dxe = np.median(np.diff(X[ei])) if len(ei) > 1 else 0.0
        dye = np.median(np.diff(Y[ei])) if len(ei) > 1 else 1.0
        az = np.arctan2(dxe, dye)
        last = float(kn[TARGET_INPUT].values[-1])
        tv = h["TVT"].to_numpy(float)
        offset = float(np.nanmean(tv[em]) - last)
        slope = 0.0
        dmd = MD[ei] - MD[ei][0]
        if len(ei) > 3 and np.ptp(dmd) > 0:
            slope = float(np.polyfit(dmd, tv[em], 1)[0] * 1000.0)
        horiz = float(np.sqrt((X[ei[-1]] - X[kn.index[-1]]) ** 2 + (Y[ei[-1]] - Y[kn.index[-1]]) ** 2))
        f = extract_level_features(w)
        rows.append(dict(offset=offset, slope=slope, az=az, az_sin=np.sin(az), az_cos=np.cos(az),
                         horiz=horiz, Xm=float(np.mean(X)), Ym=float(np.mean(Y)),
                         **{k: float(f[k]) for k in LEVEL_FEATURES}))
    return rows


def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)

def oof_ridge(X, y):
    p = np.zeros(len(y))
    for tr, va in KFold(5, shuffle=True, random_state=0).split(X):
        p[va] = Ridge(1.0).fit(X[tr], y[tr]).predict(X[va])
    return p

def oof_gbm(X, y):
    p = np.zeros(len(y))
    for tr, va in KFold(5, shuffle=True, random_state=0).split(X):
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                          min_samples_leaf=30, random_state=0)
        m.fit(X[tr], y[tr]); p[va] = m.predict(X[va])
    return p


def main():
    rows = build()
    log(f"{len(rows)} wells")
    off = np.array([r["offset"] for r in rows]); slp = np.array([r["slope"] for r in rows])
    Xb = np.array([[r[k] for k in LEVEL_FEATURES] for r in rows])
    azf = np.array([[r["az_sin"], r["az_cos"], r["horiz"], r["az_sin"]*r["v50"], r["az_cos"]*r["v50"],
                     r["az_sin"]*r["horiz"], r["az_cos"]*r["horiz"]] for r in rows])
    xy = np.array([[r["Xm"], r["Ym"]] for r in rows])
    az_only = np.array([[r["az_sin"], r["az_cos"]] for r in rows])

    print("\n=== raw corr with OFFSET ===", flush=True)
    for nm, v in [("az_sin", az_only[:,0]), ("az_cos", az_only[:,1]),
                  ("horiz", np.array([r["horiz"] for r in rows]))]:
        print(f"  {nm:10s} {np.corrcoef(v, off)[0,1]:+.3f}", flush=True)

    print("\n=== per-well OFFSET OOF R^2 ===", flush=True)
    print(f"  baseline (5 feats) Ridge      : {r2(off, oof_ridge(Xb, off)):+.3f}", flush=True)
    print(f"  + azimuth feats     Ridge     : {r2(off, oof_ridge(np.column_stack([Xb, azf]), off)):+.3f}", flush=True)
    print(f"  + azimuth + XY      GBM       : {r2(off, oof_gbm(np.column_stack([Xb, azf, xy]), off)):+.3f}", flush=True)
    print("\n=== per-well SLOPE OOF R^2 ===", flush=True)
    print(f"  baseline (5 feats) Ridge      : {r2(slp, oof_ridge(Xb, slp)):+.3f}", flush=True)
    print(f"  + azimuth feats     Ridge     : {r2(slp, oof_ridge(np.column_stack([Xb, azf]), slp)):+.3f}", flush=True)
    print(f"  + azimuth + XY      GBM       : {r2(slp, oof_gbm(np.column_stack([Xb, azf, xy]), slp)):+.3f}", flush=True)


if __name__ == "__main__":
    main()
