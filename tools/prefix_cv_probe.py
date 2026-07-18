"""Prefix-CV per-well self-calibration (notes/18, honest per-well adaptivity). For each well, hold out
the last 25% of the KNOWN prefix; measure how well the model's DRIFT assumption reconstructs it (fit
drift on the first 75%, extrapolate). Wells whose own known-zone drift is UNPREDICTABLE (high probe
error) likely have unreliable eval drift -> shrink the eval prediction toward the safe anchor. Test on
the cached OOF whether this per-well gate improves pooled RMSE (before any kernel work).

    uv run python -u tools/prefix_cv_probe.py
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from pathlib import Path
from src.data import load_split
from src.validation import score

TI = "TVT_input"


def probe_err(kt, kmd):
    """Fit a line to the first 75% of known TVT vs MD; extrapolate to the last 25%; RMSE there."""
    n = len(kt); c = int(n * 0.75)
    if c < 20 or n - c < 5:
        return 0.0
    a = np.polyfit(kmd[:c] - kmd[0], kt[:c], 1)
    pred = np.polyval(a, kmd[c:] - kmd[0])
    return float(np.sqrt(np.mean((kt[c:] - pred) ** 2)))


def main():
    d = np.load("/tmp/rogii_meta_oof.npz")
    truth, anchor, fusion, g = d["truth"], d["anchor"], d["fusion"], d["g"]
    oof_gbm, oof_mlp = d["oof_gbm"], d["oof_mlp"]
    # current best prediction (v18/v20 stack, w=0.75, 0.7/0.3 gbm/mlp)
    pred = 0.75 * (anchor + 0.7 * oof_gbm + 0.3 * oof_mlp) + 0.25 * fusion
    base = score(truth, pred)

    # per-well probe error, aligned to g (kept-well order = load_split order with the _stack_features guard)
    wells = load_split(Path("data/raw/train"))
    perr = {}
    gi = 0
    for w in wells:
        h = w.horizontal; em = w.eval_mask.to_numpy()
        kn = h[h[TI].notna()]
        if em.sum() == 0 or len(kn) < 200 or "TVT" not in h.columns:
            continue
        perr[gi] = probe_err(kn[TI].to_numpy(float), kn["MD"].to_numpy(float))
        gi += 1
    n_g = int(g.max()) + 1
    pe = np.array([perr.get(i, 0.0) for i in range(n_g)])
    row_pe = pe[g]                       # probe error per row
    print(f"kept wells {n_g} (probe map {len(perr)}); probe-err median {np.median(pe):.2f} p90 {np.percentile(pe,90):.2f}", flush=True)
    print(f"  base pooled: {base:.4f}", flush=True)

    # sanity: does probe-err correlate with per-well eval error of the model?
    well_eval_err = np.array([np.sqrt(np.mean((truth[g == i] - pred[g == i]) ** 2)) for i in range(n_g)])
    ok = np.isfinite(pe) & np.isfinite(well_eval_err)
    print(f"  corr(probe_err, eval_err) = {np.corrcoef(pe[ok], well_eval_err[ok])[0,1]:.3f}", flush=True)

    # gate: shrink pred toward anchor where probe-err is high. shrink = clip(1 - k*(pe-thr)/scale, lo, 1)
    scale = np.percentile(pe, 90) + 1e-6
    print("\n  shrink toward anchor by probe-err gate:", flush=True)
    for thr in (5.0, 8.0, 12.0):
        for k in (0.3, 0.6, 1.0):
            sh = np.clip(1.0 - k * np.maximum(row_pe - thr, 0) / scale, 0.3, 1.0)
            p2 = anchor + sh * (pred - anchor)
            print(f"    thr={thr:4.1f} k={k:.1f}: {score(truth, p2):.4f}  (base {base:.4f})", flush=True)


if __name__ == "__main__":
    main()
