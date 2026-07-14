"""Prefix-cut AUGMENTATION features: synthetic prediction problems from each well's own lateral
(PS placed at 80% of the known zone, where the well is already lateral). Produces extra (X, y)
training rows for the GBM to improve generalization to new wells. Saves /tmp/rogii_augment.npz
with X_aug, y_aug, g_aug (well id, so we can drop a well's augmented rows when it's in the val fold).

    uv run python -u _cache_augment.py
"""
from __future__ import annotations
import os, sys, time
import numpy as np
from pathlib import Path
from src.data import load_split, TARGET_INPUT, Well
from src.model import fit_level_model, level_regression, _stack_features

t0 = time.time()
def log(m): print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)
CUT = 0.80


def make_synth(well):
    h = well.horizontal
    if "TVT" not in h.columns:
        return None
    kn_pos = np.where(h[TARGET_INPUT].notna().to_numpy())[0]
    if len(kn_pos) < 400:
        return None
    cut = kn_pos[int(len(kn_pos) * CUT)]
    orig_ps = kn_pos[-1]
    h2 = h.copy(); ti = h2[TARGET_INPUT].to_numpy(float).copy()
    ti[cut:] = np.nan; h2[TARGET_INPUT] = ti
    if h2[TARGET_INPUT].notna().sum() < 200:
        return None
    # only keep the synthetic tail that is still INSIDE the original known zone (dense truth,
    # lateral) as augmentation targets — rows [cut, orig_ps]
    keep = np.zeros(len(h2), bool); keep[cut:orig_ps + 1] = True
    return Well(well.name, h2, well.typewell), keep


def main():
    wells = load_split(Path("data/raw/train"))
    lm = fit_level_model(wells)
    Xs, ys, gs = [], [], []
    for gi, w in enumerate(wells):
        r = make_synth(w)
        if r is None:
            continue
        sw, keep = r
        f = _stack_features(sw, lm)
        if f is None or "truth" not in f:
            continue
        em = sw.eval_mask.to_numpy()
        # map eval-row order to the 'keep' (known-tail) subset
        ev_idx = np.where(em)[0]
        tail = keep[ev_idx]                 # which eval rows are the synthetic known-tail
        if tail.sum() < 5:
            continue
        Xs.append(f["X"][tail]); ys.append((f["truth"] - f["anchor"])[tail])
        gs.append(np.full(int(tail.sum()), gi))
        if (gi + 1) % 150 == 0:
            log(f"  {gi+1}/{len(wells)}")
    X = np.vstack(Xs).astype(np.float32); y = np.concatenate(ys); g = np.concatenate(gs)
    np.savez_compressed("/tmp/rogii_augment.npz", X_aug=X, y_aug=y, g_aug=g)
    log(f"saved /tmp/rogii_augment.npz: {len(y)} augmented rows from {len(np.unique(g))} wells")


if __name__ == "__main__":
    main()
