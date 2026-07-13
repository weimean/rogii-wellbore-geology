"""Geology-column probe for the LEVEL — the lever I dismissed on reasoning, now measured.

The typewell has (TVT, GR, Geology=formation label). A horizontal lateral holds inside a target
formation, whose absolute TVT the typewell gives us. So "which formation is the eval zone in"
could anchor the level — and matching to a formation is robust to the quasi-periodic GR aliasing
that killed the raw-alignment levers (formations are ordered in TVT, not periodic).

ORACLE GATE first: if we knew the true target formation, how well would its typewell TVT
predict the level? Uses truth to pick the formation (upper bound). If the oracle can't beat the
5-feat anchor (offset R^2 0.098), geology is dead for level and the GR-classifier isn't worth
building. Target: offset = eval_mean_TVT - last_known.

    uv run python -u tools/geology_probe.py
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from src.data import load_split, TARGET_INPUT
from src.model import extract_level_features, LEVEL_FEATURES

t0 = time.time()
def log(m): print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


def build():
    wells = load_split(Path("data/raw/train"))
    rows = []
    for w in wells:
        h = w.horizontal; em = w.eval_mask.to_numpy()
        kn = h[h[TARGET_INPUT].notna()]
        if em.sum() < 10 or len(kn) < 200 or "TVT" not in h.columns:
            continue
        t = w.typewell
        if "Geology" not in t.columns:
            continue
        g = t["Geology"].astype(str).str.strip()
        valid = g.notna() & (g != "") & (g != "nan") & t["TVT"].notna()
        tw_tvt = t["TVT"].to_numpy(float)[valid.to_numpy()]
        tw_geo = g.to_numpy()[valid.to_numpy()]
        if len(tw_tvt) < 20:
            continue
        # per-formation median TVT (typewell reference)
        fm_med = {}
        for f in np.unique(tw_geo):
            fm_med[f] = float(np.median(tw_tvt[tw_geo == f]))
        last = float(kn[TARGET_INPUT].values[-1])
        ev_tvt = h["TVT"].to_numpy(float)[em]
        offset = float(np.nanmean(ev_tvt) - last)
        # ORACLE: which formation is each eval row truly in (nearest typewell TVT label)?
        idx = np.searchsorted(np.sort(tw_tvt), ev_tvt)
        order = np.argsort(tw_tvt); stvt = tw_tvt[order]; sgeo = tw_geo[order]
        idx = np.clip(np.searchsorted(stvt, ev_tvt), 0, len(stvt) - 1)
        ev_form = sgeo[idx]
        dom = Counter(ev_form).most_common(1)[0][0]     # dominant true formation
        geo_level = fm_med[dom]                           # its typewell median TVT
        # also: last-known's own formation median (test-available proxy = where we were pre-PS)
        li = int(np.clip(np.searchsorted(stvt, last), 0, len(stvt) - 1))
        last_form_med = fm_med[sgeo[li]]
        f = extract_level_features(w)
        rows.append(dict(offset=offset, last=last, ev_mean=float(np.nanmean(ev_tvt)),
                         geo_level=geo_level, geo_off=geo_level - last,
                         last_form_med=last_form_med,
                         **{k: float(f[k]) for k in LEVEL_FEATURES}))
    return rows


def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def oof(X, y):
    p = np.zeros(len(y))
    for tr, va in KFold(5, shuffle=True, random_state=0).split(X):
        p[va] = Ridge(1.0).fit(X[tr], y[tr]).predict(X[va])
    return p


def main():
    log("building geology table...")
    rows = build()
    y = np.array([r["offset"] for r in rows])
    ev_mean = np.array([r["ev_mean"] for r in rows])
    last = np.array([r["last"] for r in rows])
    geo_level = np.array([r["geo_level"] for r in rows])
    geo_off = np.array([r["geo_off"] for r in rows])
    Xb = np.array([[r[k] for k in LEVEL_FEATURES] for r in rows])
    log(f"{len(rows)} wells with Geology")

    print("\n=== LEVEL recovery: predict eval-mean TVT (level-RMSE, lower=better) ===", flush=True)
    print(f"  ffill (last known):        {np.sqrt(np.mean((ev_mean-last)**2)):.2f}", flush=True)
    print(f"  ORACLE geo formation median:{np.sqrt(np.mean((ev_mean-geo_level)**2)):.2f}", flush=True)
    print(f"  anchor (5-feat Ridge OOF):  {np.sqrt(np.mean((ev_mean-(last+oof(Xb,y)))**2)):.2f}", flush=True)

    # test-available formation proxy: formation of the LAST KNOWN TVT (we know last TVT at
    # inference, and the typewell Geology(TVT) map) -> its median TVT. The lateral usually holds
    # in-zone, so last-known's formation ~= eval formation.
    last_form_med = np.array([r["last_form_med"] for r in rows])
    last_form_off = last_form_med - last

    print("\n=== per-well OFFSET OOF R^2 (5-fold) ===", flush=True)
    print(f"  baseline (5 feats):            R2={r2(y, oof(Xb, y)):+.3f}", flush=True)
    print(f"  ORACLE geo_off raw:            R2={r2(y, geo_off):+.3f}  (corr={np.corrcoef(geo_off,y)[0,1]:+.3f})", flush=True)
    print(f"  baseline + ORACLE geo_off:     R2={r2(y, oof(np.column_stack([Xb, geo_off]), y)):+.3f}", flush=True)
    print(f"  TEST-AVAIL last_form_off raw:  R2={r2(y, last_form_off):+.3f}  (corr={np.corrcoef(last_form_off,y)[0,1]:+.3f})", flush=True)
    print(f"  baseline + last_form_off:      R2={r2(y, oof(np.column_stack([Xb, last_form_off]), y)):+.3f}", flush=True)


if __name__ == "__main__":
    main()
