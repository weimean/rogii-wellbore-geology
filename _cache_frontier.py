"""Cache frontier feature groups per eval row, aligned with /tmp/rogii_feats.npz:
  B: GR envelope/energy (rolling max/min/rms/std of GR at 2 scales, |dGR|)
  C: multi-scale NCC quality — corr(eval GR window, typewell GR along the HMM-predicted
     TVT path) at hw=8/15/25 (how trustworthy the alignment is locally)
Saves /tmp/rogii_frontier.npz. Live progress."""
import os, sys, time
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from src.data import load_split, TARGET_INPUT
import os as _os
_ROGII_CACHE = _os.environ.get("ROGII_CACHE", _os.path.expanduser("~/rogii_cache"))
_os.makedirs(_ROGII_CACHE, exist_ok=True)

t0 = time.time()
def log(m): print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)

# HMM predictions per row (for the NCC-quality alignment), from the base cache
D = np.load(f"{_ROGII_CACHE}/rogii_feats.npz")
HMM = D["hmm_pred"]; GID = D["gidx"]

wells = load_split(Path("data/raw/train"))
# map: for each well index, the slice of cache rows (cache built in same order/skip)
row_ptr = 0
rows = {k: [] for k in ["gmax51", "gmin51", "grms51", "gstd31", "gstd151", "gabsd",
                        "ncc8", "ncc15", "ncc25"]}

def roll(a, w, fn):
    import pandas as pd
    return getattr(pd.Series(a).rolling(w, center=True, min_periods=1), fn)().to_numpy()

for gi, wl in enumerate(wells):
    h = wl.horizontal; kn = h[h[TARGET_INPUT].notna()]; ev = h[h[TARGET_INPUT].isna()]
    em = wl.eval_mask.to_numpy()
    if em.sum() == 0 or len(kn) < 200:
        continue
    t = wl.typewell; tw_tvt = t["TVT"].to_numpy(float)
    tw_gr = gaussian_filter1d(t["GR"].ffill().bfill().to_numpy(float), 3)
    gr = h["GR"].interpolate(limit_direction="both").fillna(np.nanmean(tw_gr)).to_numpy(float)
    ei = ev.index.values
    n = len(ei)
    # --- group B: GR shape on eval rows ---
    ge = gr[ei]
    import pandas as pd
    s = pd.Series(ge)
    rows["gmax51"].append(s.rolling(51, center=True, min_periods=1).max().to_numpy())
    rows["gmin51"].append(s.rolling(51, center=True, min_periods=1).min().to_numpy())
    rows["grms51"].append(np.sqrt(pd.Series(ge**2).rolling(51, center=True, min_periods=1).mean().to_numpy()))
    rows["gstd31"].append(s.rolling(31, center=True, min_periods=1).std().fillna(0).to_numpy())
    rows["gstd151"].append(s.rolling(151, center=True, min_periods=1).std().fillna(0).to_numpy())
    rows["gabsd"].append(np.abs(np.gradient(gaussian_filter1d(ge, 5))))
    # --- group C: multi-scale NCC quality at the HMM-predicted TVT path ---
    hmm_tvt = HMM[row_ptr:row_ptr + n]     # aligned slice
    exp_gr = np.interp(hmm_tvt, tw_tvt, tw_gr)   # typewell GR along predicted path
    ge_s = gaussian_filter1d(ge, 3)
    for hw, key in [(8, "ncc8"), (15, "ncc15"), (25, "ncc25")]:
        q = np.zeros(n)
        for i in range(n):
            a0, a1 = max(0, i - hw), min(n, i + hw + 1)
            xa = ge_s[a0:a1]; xb = exp_gr[a0:a1]
            if len(xa) > 4 and xa.std() > 1e-6 and xb.std() > 1e-6:
                q[i] = np.corrcoef(xa, xb)[0, 1]
        rows[key].append(q)
    row_ptr += n
    if (gi + 1) % 100 == 0:
        log(f"  {gi+1}/{len(wells)}")

out = {k: np.concatenate(v) for k, v in rows.items()}
assert len(out["gmax51"]) == len(HMM), f"misalign {len(out['gmax51'])} vs {len(HMM)}"
np.savez_compressed(f"{_ROGII_CACHE}/rogii_frontier.npz", **out)
log(f"saved /tmp/rogii_frontier.npz, rows: {len(out['gmax51'])}")
