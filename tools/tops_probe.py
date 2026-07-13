"""Are the (train-only) formation TOPS the master key to TVT? If TVT is structurally determined
by Z + the bed-boundary depths, the whole problem reduces to 'infer the tops', and distillation
(predict structure from test-available features, tops as train labels) becomes the breakthrough
path. If not, tops don't contain the missing signal either.

Step 1: on train, how well does [Z, Z-each_top] explain TVT? (pooled + per-well R^2)
Step 2: ORACLE — does adding Z-tops to the per-well OFFSET model beat baseline (0.098)?
        (upper bound; tops are train-only so only productionizable if inferable.)

    uv run python -u tools/tops_probe.py
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from src.data import load_split, TARGET_INPUT, FORMATION_DEPTHS
from src.model import extract_level_features, LEVEL_FEATURES

t0 = time.time()
def log(m): print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)

def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def main():
    wells = load_split(Path("data/raw/train"))
    log(f"{len(wells)} wells; FORMATION_DEPTHS={FORMATION_DEPTHS}")

    # --- Step 1: does [Z, Z-tops] explain TVT? ---
    all_TVT, all_F, per_well_r2 = [], [], []
    off_rows = []
    for w in wells:
        h = w.horizontal
        if "TVT" not in h.columns or not all(c in h.columns for c in FORMATION_DEPTHS):
            continue
        em = w.eval_mask.to_numpy(); kn = h[h[TARGET_INPUT].notna()]
        if em.sum() < 10 or len(kn) < 200:
            continue
        Z = h["Z"].to_numpy(float); TVT = h["TVT"].to_numpy(float)
        tops = np.column_stack([h[c].to_numpy(float) for c in FORMATION_DEPTHS])
        F = np.column_stack([Z] + [Z - tops[:, j] for j in range(tops.shape[1])])
        ok = np.isfinite(TVT) & np.isfinite(F).all(1)
        if ok.sum() < 50:
            continue
        # per-well fit quality (structural determinism within a well)
        m = Ridge(1.0).fit(F[ok], TVT[ok])
        per_well_r2.append(r2(TVT[ok], m.predict(F[ok])))
        all_TVT.append(TVT[ok]); all_F.append(F[ok])
        # offset-relevant: mean (Z-tops) over eval rows
        last = float(kn[TARGET_INPUT].values[-1])
        emok = em & ok
        if emok.sum() > 0:
            lf = extract_level_features(w)
            off_rows.append((float(np.nanmean(TVT[emok]) - last),
                             [float(np.nanmean((Z - tops[:, j])[emok])) for j in range(tops.shape[1])],
                             [float(lf[k]) for k in LEVEL_FEATURES]))

    TVTc = np.concatenate(all_TVT); Fc = np.vstack(all_F)
    mg = Ridge(1.0).fit(Fc, TVTc)
    print("\n=== Step 1: does [Z, Z-tops] explain TVT? ===", flush=True)
    print(f"  pooled R^2 (global Ridge):     {r2(TVTc, mg.predict(Fc)):.4f}", flush=True)
    print(f"  per-well R^2 (median):         {np.median(per_well_r2):.4f}", flush=True)
    print(f"  per-well R^2 (p10..p90):       {np.percentile(per_well_r2,10):.3f} .. {np.percentile(per_well_r2,90):.3f}", flush=True)

    # --- Step 2: ORACLE offset gain from eval-mean (Z-tops) ---
    y = np.array([o for o, _, _ in off_rows])
    Xt = np.array([t for _, t, _ in off_rows])
    Xb = np.array([b for _, _, b in off_rows])
    def oof(X):
        p = np.zeros(len(y))
        for tr, va in KFold(5, shuffle=True, random_state=0).split(X):
            p[va] = Ridge(1.0).fit(X[tr], y[tr]).predict(X[va])
        return p
    print("\n=== Step 2: per-well OFFSET OOF R^2 (ORACLE — tops are train-only) ===", flush=True)
    print(f"  baseline (5 feats):            R2={r2(y, oof(Xb)):+.3f}", flush=True)
    print(f"  + ORACLE eval-mean (Z-tops):   R2={r2(y, oof(np.column_stack([Xb, Xt]))):+.3f}", flush=True)


if __name__ == "__main__":
    main()
