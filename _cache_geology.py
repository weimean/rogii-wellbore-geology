"""Cache geology LEVEL features per eval row, aligned to /tmp/rogii_feats.npz (same well order
& skip logic). The typewell Geology column is the one geological signal available at test time.

Per well (broadcast to its eval rows):
  geo_last_off  : median TVT of the formation containing the LAST KNOWN TVT, minus last (test-
                  legal; the lateral holds in-zone so last-known's formation ~= eval formation).
  geo_hmm_off   : median TVT of the formation the HMM-aligned eval TVT lands in, minus last
                  (uses hmm_pred, test-available; a GR-driven eval-formation estimate).
  geo_last_med  : that formation's median TVT (absolute level reference).
Saves /tmp/rogii_geology.npz. Run:  uv run python -u _cache_geology.py
"""
from __future__ import annotations
import os, sys, time
import numpy as np
from pathlib import Path
from src.data import load_split, TARGET_INPUT

t0 = time.time()
def log(m): print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)

D = np.load("/tmp/rogii_feats.npz")
HMM = D["hmm_pred"]; GID = D["gidx"]
wells = load_split(Path("data/raw/train"))

cols = {k: [] for k in ["geo_last_off", "geo_hmm_off", "geo_last_med"]}
row_ptr = 0
for gi, w in enumerate(wells):
    h = w.horizontal; em = w.eval_mask.to_numpy()
    kn = h[h[TARGET_INPUT].notna()]
    if em.sum() == 0 or len(kn) < 200:
        continue
    n = int(em.sum())
    t = w.typewell
    last = float(kn[TARGET_INPUT].values[-1])
    hmm_tvt = HMM[row_ptr:row_ptr + n]
    if "Geology" not in t.columns:
        cols["geo_last_off"].append(np.zeros(n)); cols["geo_hmm_off"].append(np.zeros(n))
        cols["geo_last_med"].append(np.full(n, last))
        row_ptr += n; continue
    g = t["Geology"].astype(str).str.strip()
    valid = (g.notna() & (g != "") & (g != "nan") & t["TVT"].notna()).to_numpy()
    tw_tvt = t["TVT"].to_numpy(float)[valid]; tw_geo = g.to_numpy()[valid]
    order = np.argsort(tw_tvt); stvt = tw_tvt[order]; sgeo = tw_geo[order]
    fm_med = {f: float(np.median(tw_tvt[tw_geo == f])) for f in np.unique(tw_geo)}

    def form_med(tvt_val):
        i = int(np.clip(np.searchsorted(stvt, tvt_val), 0, len(stvt) - 1))
        return fm_med[sgeo[i]]

    last_med = form_med(last)
    hmm_med = np.array([form_med(v) for v in hmm_tvt])   # per-row HMM-formation median
    cols["geo_last_off"].append(np.full(n, last_med - last))
    cols["geo_hmm_off"].append(hmm_med - last)
    cols["geo_last_med"].append(np.full(n, last_med))
    row_ptr += n
    if (gi + 1) % 150 == 0:
        log(f"  {gi+1}/{len(wells)}")

out = {k: np.concatenate(v) for k, v in cols.items()}
assert len(out["geo_last_off"]) == len(HMM), f"misalign {len(out['geo_last_off'])} vs {len(HMM)}"
np.savez_compressed("/tmp/rogii_geology.npz", **out)
log(f"saved /tmp/rogii_geology.npz rows={len(out['geo_last_off'])}")
