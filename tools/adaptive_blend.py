"""Adaptive (well-level) blend: instead of a global w, trust the HMM/PF fusion where it's
reliable and drop it where it isn't. Motivated by residual analysis (error/tail dominated by
long, high-uncertainty wells where the fusion drifts). Free: post-hoc on the cached OOF.

  w_well = clip(w0 + s1*z(mean hmm_std) + s2*z(eval_len) + s3*z(mean |hmm-pf|), lo, hi)
  pred   = w_well*(anchor+gbm) + (1-w_well)*fusion

Tune (w0,s1,s2,s3) on dev; evaluate lockbox + pooled. Compare to global w=0.75.

    uv run python -u tools/adaptive_blend.py
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
    first, counts = s["first"], s["counts"]
    pred = np.load("/tmp/rogii_oof.npz")["pred"]
    gbm = (pred - 0.4 * fusion) / 0.6 - anchor        # OOF residual (from the w=0.6 cache)
    corr = anchor + gbm
    d = np.load("/tmp/rogii_feats.npz"); pf = np.load("/tmp/rogii_pf.npz")
    uniq = np.unique(g)
    # per-well uncertainty signals, z-scored, broadcast to rows
    def wellmean(a): return np.repeat([a[first[i]:first[i]+counts[i]].mean() for i in range(len(uniq))], counts)
    hs = wellmean(d["hmm_std"]); el = np.repeat(counts.astype(float), counts)
    dis = wellmean(np.abs(d["hmm_pred"] - pf["pf_pred"]))
    def z(a): return (a - a.mean()) / (a.std() + 1e-9)
    zhs, zel, zdis = z(hs), z(el), z(dis)

    rng = np.random.default_rng(42)
    lock_wells = set(rng.choice(uniq, int(len(uniq) * 0.2), replace=False).tolist())
    dev = np.array([gi not in lock_wells for gi in g]); lock = ~dev

    def blend(w0, s1, s2, s3, lo=0.55, hi=1.0):
        w = np.clip(w0 + s1 * zhs + s2 * zel + s3 * zdis, lo, hi)
        return w * corr + (1 - w) * fusion

    base = score(truth[dev], (0.75 * corr + 0.25 * fusion)[dev])
    print(f"  global w=0.75  dev {base:.4f}  lock {score(truth[lock], (0.75*corr+0.25*fusion)[lock]):.4f}", flush=True)
    best = (base, None)
    for w0 in (0.70, 0.75, 0.80):
        for s1 in (0.0, 0.05, 0.1, 0.15):
            for s2 in (0.0, 0.05, 0.1, 0.15):
                for s3 in (0.0, 0.05, 0.1):
                    p = blend(w0, s1, s2, s3)
                    dv = score(truth[dev], p[dev])
                    if dv < best[0]:
                        best = (dv, (w0, s1, s2, s3))
    w0, s1, s2, s3 = best[1]
    p = blend(w0, s1, s2, s3)
    print(f"\n  adaptive best (dev): w0={w0} s_hmmstd={s1} s_evallen={s2} s_disagree={s3}", flush=True)
    print(f"    dev {score(truth[dev],p[dev]):.4f}  lock {score(truth[lock],p[lock]):.4f}  pooled {score(truth,p):.4f}", flush=True)
    print(f"    vs global 0.75: pooled {score(truth, 0.75*corr+0.25*fusion):.4f}", flush=True)


if __name__ == "__main__":
    main()
