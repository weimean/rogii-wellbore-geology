"""Whole-pipeline optics #3 — PREFIX CV (the task-aligned validation the winners use).
Place a synthetic prediction-start at 65% of each well's KNOWN zone (mask TVT_input from there
on), re-run the full pipeline on these synthetic-PS wells, and predict/score the masked tail
(dense known-zone truth). Then sweep the final blend w on this metric — if it agrees the
GroupKFold-tuned w=0.70 is right, that's strong confirmation; if not, a mistuning to fix.

    uv run python -u tools/prefix_cv.py
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
import lightgbm as lgb
from catboost import CatBoostRegressor
from src.data import load_split, TARGET_INPUT, Well
from src.model import fit_level_model, level_regression, _stack_features
from src.validation import score

t0 = time.time()
def log(m): print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)

CUT = 0.65


def make_synth(well):
    """Return a synthetic-PS well (TVT_input masked from 65% of the known zone) or None."""
    h = well.horizontal
    if "TVT" not in h.columns:
        return None
    kn_pos = np.where(h[TARGET_INPUT].notna().to_numpy())[0]
    if len(kn_pos) < 300:
        return None
    cut = kn_pos[int(len(kn_pos) * CUT)]
    h2 = h.copy()
    ti = h2[TARGET_INPUT].to_numpy(float).copy()
    ti[cut:] = np.nan                       # synthetic PS at the cut
    h2[TARGET_INPUT] = ti
    if h2[TARGET_INPUT].notna().sum() < 200 or (~np.isfinite(ti)).sum() < 10:
        return None
    return Well(well.name, h2, well.typewell)


def main():
    wells = load_split(Path("data/raw/train"))
    synth = [(w.name, make_synth(w)) for w in wells]
    synth = [(n, s) for n, s in synth if s is not None]
    names = np.array([n for n, _ in synth]); sw = [s for _, s in synth]
    log(f"{len(sw)} synthetic-PS wells; OOF level anchor...")
    gkf = GroupKFold(5)

    oof_anchor = {}
    for tr, va in gkf.split(sw, groups=names):
        lm = fit_level_model([sw[i] for i in tr])
        for i in va:
            w = sw[i]; em = w.eval_mask.to_numpy()
            if em.sum():
                oof_anchor[w.name] = level_regression(w, lm)[em]

    log("stack features on synthetic wells (HMM+PF)...")
    rows = []
    lm_full = fit_level_model(sw)
    for k, w in enumerate(sw):
        if w.name not in oof_anchor:
            continue
        f = _stack_features(w, lm_full, anchor_override=oof_anchor[w.name])
        if f is None or "truth" not in f:
            continue
        rows.append(f)
        if (k + 1) % 150 == 0:
            log(f"  {k+1}/{len(sw)}")

    g = np.concatenate([np.full(len(f["truth"]), i) for i, f in enumerate(rows)])
    truth = np.concatenate([f["truth"] for f in rows])
    anchor = np.concatenate([f["anchor"] for f in rows])
    fusion = np.concatenate([f["fusion"] for f in rows])
    X = np.vstack([f["X"] for f in rows]); y = truth - anchor
    idx_in = np.concatenate([np.arange((g == i).sum()) for i in range(g.max() + 1)])
    sub = (idx_in % 5) == 0

    log("OOF stack (seed-ens LGB+Cat, step=5)...")
    oof = np.zeros(len(y))
    for tr, va in gkf.split(X, groups=g):
        ti = tr[sub[tr]]; ps = []
        for sd in (0, 1, 2):
            ps.append(lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=63,
                      min_child_samples=200, reg_lambda=1.0, subsample=0.8, subsample_freq=1,
                      colsample_bytree=0.8, random_state=sd, n_jobs=-1, verbose=-1).fit(X[ti], y[ti]).predict(X[va]))
            ps.append(CatBoostRegressor(iterations=600, learning_rate=0.03, depth=8, l2_leaf_reg=3.0,
                      rsm=0.8, random_seed=sd, thread_count=-1, verbose=0).fit(X[ti], y[ti]).predict(X[va]))
        oof[va] = np.mean(ps, axis=0)
    corr = anchor + oof

    print("\n=== PREFIX CV (synthetic PS @65% known) — final blend w sweep ===", flush=True)
    best = (1e9, None)
    for w in np.round(np.arange(0.40, 0.96, 0.05), 2):
        r = score(truth, w * corr + (1 - w) * fusion)
        tag = " <- current" if abs(w - 0.70) < 1e-6 else ""
        if r < best[0]:
            best = (r, w)
        print(f"  w={w:.2f}  RMSE={r:.4f}{tag}", flush=True)
    print(f"\n  prefix-CV best w = {best[1]:.2f} (RMSE {best[0]:.4f}); production w=0.70", flush=True)


if __name__ == "__main__":
    main()
