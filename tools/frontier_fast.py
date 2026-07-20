"""Fast stack-experiment harness. Two speedups over the ad-hoc scripts:

1. Vectorized build (cache rows are grouped & sorted by well -> use reduceat/repeat instead
   of per-well `g==u` masks) + the assembled matrix cached to /tmp/rogii_stackX.npz.
2. Per-well TRAIN subsampling: fit on 1/step of eval rows (the GBM correction is smooth so CV
   barely moves) but always PREDICT/score at full resolution. ~step x faster iteration.

    from tools.frontier_fast import load_stack, oof, train_submask
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from sklearn.model_selection import GroupKFold, KFold
from sklearn.linear_model import Ridge
import os as _os
_ROGII_CACHE = _os.environ.get("ROGII_CACHE", _os.path.expanduser("~/rogii_cache"))
_os.makedirs(_ROGII_CACHE, exist_ok=True)

STACK_FEATS = ["hmm_pred", "hmm_std", "z_from_ps", "md_from_ps", "horiz_from_ps", "dz",
               "tort", "gr50", "gr_dev", "az_sin", "az_cos", "eval_frac",
               "v50", "v200", "gr_off", "known_range", "eval_len"]
LVL = ["v50", "v200", "gr_off", "eval_len", "known_range"]
CACHE = f"{_ROGII_CACHE}/rogii_stackX.npz"


def load_stack(rebuild=False):
    """Return dict with X, truth, anchor, fusion, g, first, counts (cached)."""
    if os.path.exists(CACHE) and not rebuild:
        z = np.load(CACHE)
        return {k: z[k] for k in z.files}
    d = np.load(f"{_ROGII_CACHE}/rogii_feats.npz"); pf = np.load(f"{_ROGII_CACHE}/rogii_pf.npz")
    fr = np.load(f"{_ROGII_CACHE}/rogii_frontier.npz")
    g = d["gidx"]
    assert np.all(np.diff(g) >= 0), "cache rows must be grouped/sorted by well"
    truth, last = d["truth"], d["last_tvt"]
    hp, hs = d["hmm_pred"], d["hmm_std"]; pp, ps = pf["pf_pred"], pf["pf_std"]
    uniq, first = np.unique(g, return_index=True)
    counts = np.diff(np.append(first, len(g)))

    # per-well OOF level anchor (Ridge on offset), expanded back to rows via repeat
    well_X = np.column_stack([d[k][first] for k in LVL])
    well_off = np.add.reduceat(truth, first) / counts - last[first]
    anchor_off = np.zeros(len(uniq))
    for tr, va in KFold(5, shuffle=True, random_state=0).split(well_X):
        anchor_off[va] = Ridge(1.0).fit(well_X[tr], well_off[tr]).predict(well_X[va])
    anchor = last + np.repeat(anchor_off, counts)

    wt = 1 / (1 + (hs / 4) ** 2); fusion = wt * hp + (1 - wt) * anchor
    wtp = 1 / (1 + (ps / 4) ** 2); pf_fus = wtp * pp + (1 - wtp) * anchor
    A = np.column_stack([np.abs(hp - pp), hs + ps, np.abs(hp - pp) / (hs + ps + 1),
                         np.abs(hp - anchor), np.abs(pp - anchor)])
    B = np.column_stack([fr[k] for k in ["gmax51", "gmin51", "grms51", "gstd31", "gstd151", "gabsd"]])
    X = np.column_stack([d[k] for k in STACK_FEATS] +
                        [anchor, hp - anchor, fusion, wt, pp, ps, pp - anchor, hp - pp, pf_fus])
    X = np.column_stack([X, A, B]).astype(np.float32)
    out = dict(X=X, truth=truth, anchor=anchor, fusion=fusion, g=g, first=first, counts=counts)
    np.savez_compressed(CACHE, **out)
    return out


def train_submask(g, first, counts, step):
    """Keep every `step`-th eval row within each well (evenly)."""
    idx_in_well = np.arange(len(g)) - np.repeat(first, counts)
    return (idx_in_well % step) == 0


def oof(make_model, X, y, g, train_mask=None):
    """5-fold GroupKFold OOF. Fit on train rows (optionally subsampled by train_mask),
    predict on ALL validation rows. Returns full-length prediction array."""
    p = np.zeros(len(y))
    for tr, va in GroupKFold(5).split(X, groups=g):
        ti = tr if train_mask is None else tr[train_mask[tr]]
        m = make_model(); m.fit(X[ti], y[ti]); p[va] = m.predict(X[va])
    return p
