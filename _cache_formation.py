"""Rich spatial formation features (top-team FormationPlaneKNN + DenseANCCImputer), aligned to
/tmp/rogii_feats.npz rows. Faithful port of the LB-7 recipe (notes/14).

Per eval row emits: tvtF_{6} (b_full), tvtFw_{6} (b_wls), tvtF50_{6} (b_late), tvt_dense, plane_knn_d,
dense_dist, dense_std  -> 22 features. Self-excluded (LOWO). Reference from train wells only.

    uv run python -u _cache_formation.py
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from pathlib import Path
from scipy.spatial import cKDTree
from src.data import load_split
from src.model import _stack_features

t0 = time.time()
def log(m): print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)
FORMS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]
PLANE_K = 10; DENSE_SPW = 60; DENSE_K = 20


def seg_b_well(ktvt, kz, form_col):
    bv = ktvt + kz - form_col; n = len(bv); b_full = float(np.median(bv))
    b_late = float(np.median(bv[max(0, n - 50):])) if n >= 5 else b_full
    w = np.exp(0.02 * np.arange(n)); w /= w.sum()
    return b_full, b_late, float(np.dot(w, bv))


class FormationPlaneKNN:
    def __init__(self, wells):
        rows = []
        for w in wells:
            h = w.horizontal
            if not all(c in h.columns for c in FORMS):
                continue
            df = h[["X", "Y"] + FORMS].dropna()
            if len(df) == 0:
                continue
            row = {"wid": w.name, "x": float(df.X.median()), "y": float(df.Y.median())}
            for c in FORMS:
                row[f"{c}_m"] = float(df[c].median())
            rows.append(row)
        self.df = pd.DataFrame(rows); self.wmap = {w: i for i, w in enumerate(self.df.wid)}
        xy = self.df[["x", "y"]].to_numpy(); self.scale = np.where(xy.std(0) < 1e-3, 1., xy.std(0))
        self.tree = cKDTree(xy / self.scale); self.xa = self.df.x.to_numpy(); self.ya = self.df.y.to_numpy()
        self.fa = self.df[[f"{c}_m" for c in FORMS]].to_numpy(np.float64)

    def impute(self, xy_q, self_wid=None, k=PLANE_K):
        q = xy_q / self.scale; nf = min(k + 5, len(self.df))
        dist, idx = self.tree.query(q, k=nf, workers=-1)
        if self_wid in self.wmap:
            dist = np.where(idx == self.wmap[self_wid], np.inf, dist)
        ordr = np.argpartition(dist, min(k - 1, nf - 1), 1)[:, :k]
        dk = np.take_along_axis(dist, ordr, 1); ik = np.take_along_axis(idx, ordr, 1)
        vk = np.isfinite(dk); w = np.where(vk, 1. / (dk + 1e-3), 0.).astype(np.float64)
        xn = self.xa[ik]; yn = self.ya[ik]; fn = self.fa[ik]; wx = w * xn; wy = w * yn
        A = np.zeros((len(q), 3, 3))
        A[:, 0, 0] = (wx * xn).sum(1); A[:, 0, 1] = (wx * yn).sum(1); A[:, 0, 2] = wx.sum(1)
        A[:, 1, 0] = A[:, 0, 1]; A[:, 1, 1] = (wy * yn).sum(1); A[:, 1, 2] = wy.sum(1)
        A[:, 2, 0] = A[:, 0, 2]; A[:, 2, 1] = A[:, 1, 2]; A[:, 2, 2] = w.sum(1)
        A[:, 0, 0] += 1e-9; A[:, 1, 1] += 1e-9; A[:, 2, 2] += 1e-9
        rhs = np.stack([(wx[:, :, None] * fn).sum(1), (wy[:, :, None] * fn).sum(1), (w[:, :, None] * fn).sum(1)], 1)
        try:
            coef = np.linalg.solve(A, rhs)
        except Exception:
            coef = np.zeros((len(q), 3, 6))
            for r in range(len(q)):
                try: coef[r] = np.linalg.pinv(A[r]) @ rhs[r]
                except Exception: pass
        Xq = xy_q[:, 0]; Yq = xy_q[:, 1]
        pred = (Xq[:, None] * coef[:, 0, :] + Yq[:, None] * coef[:, 1, :] + coef[:, 2, :]).astype(np.float32)
        pred[~vk.any(1)] = self.fa.mean(0)
        return pred, np.where(vk, dk, np.inf).min(1).astype(np.float32)


class DenseANCCImputer:
    def __init__(self, wells, spw=DENSE_SPW):
        xs, ys, an, wd = [], [], [], []
        for w in wells:
            h = w.horizontal
            if "ANCC" not in h.columns:
                continue
            df = h[["X", "Y", "ANCC"]].dropna()
            if len(df) == 0:
                continue
            ix = np.linspace(0, len(df) - 1, min(spw, len(df)), dtype=int); s = df.iloc[ix]
            xs.append(s.X.values); ys.append(s.Y.values); an.append(s.ANCC.values); wd.extend([w.name] * len(s))
        self.xy = np.column_stack([np.concatenate(xs), np.concatenate(ys)])
        self.ancc = np.concatenate(an).astype(np.float32); self.wids = np.array(wd)
        self.scale = np.where(self.xy.std(0) < 1e-3, 1., self.xy.std(0)); self.tree = cKDTree(self.xy / self.scale)

    def impute(self, xy_q, self_wid=None, k=DENSE_K, nfetch=5000):
        xy_q = np.atleast_2d(xy_q); q = xy_q / self.scale; nf = min(nfetch, len(self.ancc))
        dist, idx = self.tree.query(q, k=nf, workers=-1)
        if self_wid:
            dist = np.where(self.wids[idx] == self_wid, np.inf, dist)
        ordr = np.argpartition(dist, min(k - 1, nf - 1), 1)[:, :k]
        dk = np.take_along_axis(dist, ordr, 1); ik = np.take_along_axis(idx, ordr, 1)
        vk = np.isfinite(dk); w = np.where(vk, 1. / (dk + 1e-3), 0.); sw = w.sum(1); safe = np.where(sw < 1e-9, 1., sw)
        a = self.ancc[ik]; ap = (a * w).sum(1) / safe; ap = np.where(sw < 1e-9, float(self.ancc.mean()), ap)
        var = ((a - ap[:, None]) ** 2 * w).sum(1) / safe
        return ap.astype(np.float32), np.sqrt(np.maximum(var, 0.)).astype(np.float32), np.where(vk, dk, np.inf).min(1).astype(np.float32)


def main():
    wells = load_split(Path("data/raw/train"))
    FI = FormationPlaneKNN(wells); DI = DenseANCCImputer(wells)
    log("imputers built; iterating wells")
    names = [f"tvtF_{c}" for c in FORMS] + [f"tvtFw_{c}" for c in FORMS] + [f"tvtF50_{c}" for c in FORMS] \
            + ["tvt_dense", "plane_knn_d", "dense_dist", "dense_std"]
    out = {n: [] for n in names}
    gi = 0
    for k, w in enumerate(wells):
        em = w.eval_mask.to_numpy()
        if em.sum() == 0:
            continue
        n = int(em.sum())
        f = _stack_features(w, None, anchor_override=np.zeros(n))
        if f is None or "truth" not in f:
            continue
        h = w.horizontal
        Z = h["Z"].to_numpy(); T = h["TVT"].to_numpy() if "TVT" in h.columns else np.full(len(Z), np.nan)
        Ti = h["TVT_input"].to_numpy()
        ev = (~np.isfinite(Ti)); pp = np.isfinite(Ti)
        eidx = np.where(em)[0]                      # eval rows (== TVT_input NaN tail)
        kidx = np.where(pp)[0]
        xy_e = h[["X", "Y"]].to_numpy()[eidx]; xy_k = h[["X", "Y"]].to_numpy()[kidx]
        z_e = Z[eidx]; ktvt = Ti[kidx]; z_k = Z[kidx]
        swid = w.name
        form_e, knn_d = FI.impute(xy_e, self_wid=swid); form_k, _ = FI.impute(xy_k, self_wid=swid)
        for fi, c in enumerate(FORMS):
            bf, bl, bw = seg_b_well(ktvt, z_k, form_k[:, fi])
            out[f"tvtF_{c}"].append((-z_e + form_e[:, fi] + bf).astype(np.float32))
            out[f"tvtFw_{c}"].append((-z_e + form_e[:, fi] + bw).astype(np.float32))
            out[f"tvtF50_{c}"].append((-z_e + form_e[:, fi] + bl).astype(np.float32))
        d_e, dstd_e, d_dist = DI.impute(xy_e, self_wid=swid); d_k, _, _ = DI.impute(xy_k, self_wid=swid)
        bfd, _, _ = seg_b_well(ktvt, z_k, d_k)
        out["tvt_dense"].append((-z_e + d_e + bfd).astype(np.float32))
        out["plane_knn_d"].append(np.log1p(knn_d).astype(np.float32))
        out["dense_dist"].append(np.log1p(d_dist).astype(np.float32))
        out["dense_std"].append(dstd_e.astype(np.float32))
        gi += 1
        if (k + 1) % 150 == 0:
            log(f"  {k+1}/{len(wells)} ({gi} kept)")
    out = {kk: np.concatenate(v) for kk, v in out.items()}
    ref = np.load("/tmp/rogii_feats.npz")["truth"]
    assert len(out["tvtF_ANCC"]) == len(ref), f"align mismatch {len(out['tvtF_ANCC'])} vs {len(ref)}"
    np.savez_compressed("/tmp/rogii_formation.npz", **out)
    log(f"saved {len(out['tvtF_ANCC'])} rows, {len(names)} feats -> /tmp/rogii_formation.npz")


if __name__ == "__main__":
    main()
