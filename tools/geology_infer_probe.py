"""Can we INFER the (train-only) typewell Geology from test-available data, and does the
inferred formation keep the LEVEL signal? Fully test-legal: classifier trained on train
typewells, applied to a held-out well's typewell (GR + TVT only).

Validation mirrors inference: to score well W we infer W's typewell Geology from a classifier
trained on OTHER wells' typewells (GroupKFold), then rebuild the geology offset feature from the
INFERRED labels — never W's own labels.

Steps:
  1) GATE: cross-well formation-classification accuracy (GR + shape + normalized TVT position).
  2) SIGNAL: offset R^2 from baseline vs +inferred-geo vs +true-geo(0.187) vs baseline(0.098).

    uv run python -u tools/geology_infer_probe.py
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import KFold, GroupKFold
from sklearn.linear_model import Ridge
import lightgbm as lgb
from src.data import load_split, TARGET_INPUT
from src.model import extract_level_features, LEVEL_FEATURES

t0 = time.time()
def log(m): print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


def tw_features(tvt, gr):
    """Per-typewell-row features for formation classification (test-legal: TVT+GR only)."""
    gr = gaussian_filter1d(gr, 2)
    pos = (tvt - tvt.min()) / (tvt.max() - tvt.min() + 1e-9)   # normalized depth position
    def roll(w, fn):
        import pandas as pd
        return getattr(pd.Series(gr).rolling(w, center=True, min_periods=1), fn)().to_numpy()
    return np.column_stack([gr, pos, roll(15, "mean"), roll(15, "std"),
                            roll(51, "mean"), np.gradient(gr)])


def build():
    wells = load_split(Path("data/raw/train"))
    per_well = []   # (name, tw_tvt, tw_gr, tw_geo, last, offset, level_feats)
    for w in wells:
        h = w.horizontal; em = w.eval_mask.to_numpy()
        kn = h[h[TARGET_INPUT].notna()]
        t = w.typewell
        if em.sum() < 10 or len(kn) < 200 or "TVT" not in h.columns or "Geology" not in t.columns:
            continue
        g = t["Geology"].astype(str).str.strip()
        valid = (g.notna() & (g != "") & (g != "nan") & t["TVT"].notna()).to_numpy()
        if valid.sum() < 30:
            continue
        tw_tvt = t["TVT"].to_numpy(float)[valid]; tw_gr = t["GR"].to_numpy(float)[valid]
        tw_geo = g.to_numpy()[valid]
        last = float(kn[TARGET_INPUT].values[-1])
        offset = float(np.nanmean(h["TVT"].to_numpy(float)[em]) - last)
        lf = extract_level_features(w)
        per_well.append((w.name, tw_tvt, tw_gr, tw_geo, last, offset,
                         [float(lf[k]) for k in LEVEL_FEATURES]))
    return per_well


def form_med_map(tvt, geo):
    return {f: float(np.median(tvt[geo == f])) for f in np.unique(geo)}


def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def main():
    log("building typewell tables...")
    pw = build()
    n = len(pw)
    log(f"{n} wells with train Geology")

    # global label set
    labels = sorted(set(l for _, _, _, geo, _, _, _ in pw for l in np.unique(geo)))
    lab_idx = {l: i for i, l in enumerate(labels)}
    log(f"{len(labels)} formations: {labels}")

    # cross-well OOF: infer each well's typewell Geology from a classifier trained on OTHER wells
    gk = GroupKFold(5)
    # stack all typewell rows with a well-group id
    Xrows, yrows, grp = [], [], []
    for i in range(n):
        _, tvt, gr, geo, last, off, lf = pw[i]
        F = tw_features(tvt, gr)
        Xrows.append(F); yrows.append([lab_idx[l] for l in geo]); grp.append(np.full(len(F), i))
    Xall = np.vstack(Xrows); yall = np.concatenate(yrows); gall = np.concatenate(grp)

    inferred = [None] * n
    accs = []
    for tr, va in gk.split(Xall, groups=gall):
        clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63,
                                 min_child_samples=100, random_state=0, n_jobs=-1, verbose=-1)
        clf.fit(Xall[tr], yall[tr])
        pred = clf.predict(Xall[va])
        accs.append(float((pred == yall[va]).mean()))
        for wi in np.unique(gall[va]):
            m = gall == wi
            # rows of this well within va
            inferred[wi] = clf.predict(Xall[m])
    log(f"cross-well formation accuracy: {np.mean(accs):.3f}")

    # build offset targets + features
    y = np.array([off for _, _, _, _, _, off, _ in pw])
    Xb = np.array([lf for *_, lf in pw])
    true_off = np.zeros(n); inf_off = np.zeros(n)
    inv_lab = {i: l for l, i in lab_idx.items()}
    for i, (_, tvt, gr, geo, last, off, _) in enumerate(pw):
        tmed = form_med_map(tvt, geo)
        j = int(np.clip(np.searchsorted(np.sort(tvt), last), 0, len(tvt) - 1))
        order = np.argsort(tvt); stvt = tvt[order]; sgeo = geo[order]
        true_off[i] = tmed[sgeo[int(np.clip(np.searchsorted(stvt, last), 0, len(stvt)-1))]] - last
        # inferred: formation medians from INFERRED labels on this well's typewell
        inf_geo = np.array([inv_lab[k] for k in inferred[i]])
        imed = form_med_map(tvt, inf_geo)
        sgeo_i = inf_geo[order]
        inf_off[i] = imed[sgeo_i[int(np.clip(np.searchsorted(stvt, last), 0, len(stvt)-1))]] - last

    def oof(X):
        p = np.zeros(n)
        for tr, va in KFold(5, shuffle=True, random_state=0).split(X):
            p[va] = Ridge(1.0).fit(X[tr], y[tr]).predict(X[va])
        return p

    print("\n=== per-well OFFSET OOF R^2 ===", flush=True)
    print(f"  baseline (5 feats):        R2={r2(y, oof(Xb)):+.3f}", flush=True)
    print(f"  + TRUE-geo off:            R2={r2(y, oof(np.column_stack([Xb, true_off]))):+.3f}  (corr {np.corrcoef(true_off,y)[0,1]:+.3f})", flush=True)
    print(f"  + INFERRED-geo off (legal):R2={r2(y, oof(np.column_stack([Xb, inf_off]))):+.3f}  (corr {np.corrcoef(inf_off,y)[0,1]:+.3f})", flush=True)


if __name__ == "__main__":
    main()
