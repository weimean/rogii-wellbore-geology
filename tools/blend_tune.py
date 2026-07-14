"""Whole-pipeline optics #1 — re-tune the final blend weight (gbm-corrected anchor vs fusion).
Free: back the GBM residual out of the cached OOF and sweep w without retraining. Tune on dev,
evaluate on lockbox + pooled. Current final = 0.6*(anchor+gbm) + 0.4*fusion.

    uv run python -u tools/blend_tune.py
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from tools.frontier_fast import load_stack
from src.validation import score


def main():
    s = load_stack()
    anchor, fusion, truth, g = s["anchor"], s["fusion"], s["truth"], s["g"]
    pred = np.load("/tmp/rogii_oof.npz")["pred"]        # = 0.6*(anchor+gbm) + 0.4*fusion
    gbm = (pred - 0.4 * fusion) / 0.6 - anchor          # recover the OOF residual prediction
    corr = anchor + gbm                                 # gbm-corrected anchor

    uniq = np.unique(g); rng = np.random.default_rng(42)
    lock_wells = set(rng.choice(uniq, int(len(uniq) * 0.2), replace=False).tolist())
    dev = np.array([gi not in lock_wells for gi in g]); lock = ~dev

    def blended(w): return w * corr + (1 - w) * fusion
    print("  w      dev      lock     pooled", flush=True)
    best = (1e9, None)
    for w in np.round(np.arange(0.30, 0.96, 0.05), 2):
        p = blended(w)
        d = score(truth[dev], p[dev]); l = score(truth[lock], p[lock]); a = score(truth, p)
        flag = " <- current" if abs(w - 0.6) < 1e-6 else ("  * dev-best" if d < best[0] else "")
        if d < best[0]:
            best = (d, w)
        print(f"  {w:.2f}  {d:.4f}  {l:.4f}  {a:.4f}{flag}", flush=True)
    bw = best[1]
    p0, pb = blended(0.6), blended(bw)
    print(f"\n  current w=0.60 : lock {score(truth[lock],p0[lock]):.4f}  pooled {score(truth,p0):.4f}", flush=True)
    print(f"  dev-best w={bw:.2f}: lock {score(truth[lock],pb[lock]):.4f}  pooled {score(truth,pb):.4f}", flush=True)


if __name__ == "__main__":
    main()
