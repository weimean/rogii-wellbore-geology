# ROGII Wellbore Geology — hmm_fusion submission (self-contained)
# 2nd-order (TVT, dip-rate) forward-backward HMM fused with a Ridge level model
# by posterior variance. CV RMSE 14.24 (773-well GroupKFold), -10.5% vs ffill.
from __future__ import annotations
import glob, os
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import Ridge

TARGET_INPUT = "TVT_input"

# ---- locate competition data (robust to mount path) -------------------------
def _find_split_dirs():
    hits = glob.glob("/kaggle/input/**/*__horizontal_well.csv", recursive=True)
    if not hits:
        hits = glob.glob("**/*__horizontal_well.csv", recursive=True)  # local
    dirs = sorted(set(os.path.dirname(h) for h in hits))
    train_dir = test_dir = None
    for d in dirs:
        f = sorted(glob.glob(os.path.join(d, "*__horizontal_well.csv")))[0]
        cols = pd.read_csv(f, nrows=1).columns
        if "TVT" in cols:
            train_dir = d
        else:
            test_dir = d
    return train_dir, test_dir

_train, _test = _find_split_dirs()
print("input listing:", glob.glob("/kaggle/input/*"))
print("train dir:", _train, "| test dir:", _test)
assert _train is not None and _test is not None, "could not find train/test well CSVs"
TRAIN_DIR = Path(_train)
TEST_DIR = Path(_test)


@dataclass
class Well:
    name: str
    horizontal: pd.DataFrame
    typewell: pd.DataFrame

    @property
    def eval_mask(self) -> pd.Series:
        return self.horizontal[TARGET_INPUT].isna()


def well_names(split_dir: Path):
    return sorted(p.name.split("__")[0] for p in split_dir.glob("*__horizontal_well.csv"))


def load_well(split_dir: Path, name: str) -> Well:
    h = pd.read_csv(split_dir / f"{name}__horizontal_well.csv")
    t = pd.read_csv(split_dir / f"{name}__typewell.csv")
    return Well(name, h, t)


def load_split(split_dir: Path):
    return [load_well(split_dir, n) for n in well_names(split_dir)]


def submission_id(well: str, row_index: int) -> str:
    return f"{well}_{row_index}"


# ---- helpers ----------------------------------------------------------------
def baseline_ffill(h: pd.DataFrame) -> np.ndarray:
    return h[TARGET_INPUT].ffill().bfill().to_numpy().astype(float)


def _calibrate_gr(known, tw_gr_fn):
    kg = known.dropna(subset=["GR", TARGET_INPUT])
    if len(kg) < 50:
        return 1.0, 0.0
    tw_at = tw_gr_fn(kg[TARGET_INPUT].values)
    valid = ~np.isnan(tw_at)
    if valid.sum() < 20:
        return 1.0, 0.0
    return tuple(np.polyfit(tw_at[valid], kg["GR"].values[valid], 1))


def _smooth_gr(h, sigma):
    gr = h["GR"].values.copy()
    valid = ~np.isnan(gr)
    if valid.sum() < 20:
        return np.full(len(gr), np.nan)
    gi = np.interp(np.arange(len(gr)), np.where(valid)[0], gr[valid])
    return gaussian_filter1d(gi, sigma=sigma)


def _gr_offset_raw(well, gr_sigma=100.0, max_offset=30.0):
    h, t = well.horizontal, well.typewell
    known = h[h[TARGET_INPUT].notna()]; evz = h[h[TARGET_INPUT].isna()]
    if len(known) == 0 or len(evz) == 0:
        return 0.0
    last = float(known[TARGET_INPUT].iloc[-1])
    tw_tvt = t["TVT"].values
    tw_sm = gaussian_filter1d(t["GR"].values, sigma=3)
    tw_fn = interp1d(tw_tvt, tw_sm, bounds_error=False, fill_value=np.nan)
    a, b = _calibrate_gr(known, tw_fn)
    gs = _smooth_gr(h, gr_sigma)
    if np.all(np.isnan(gs)):
        return 0.0
    egm = gs[evz.index.values].mean()
    best_m, best_o = np.inf, 0.0
    for o in np.linspace(-max_offset, max_offset, 121):
        c = tw_fn(last + o)
        if np.isnan(c):
            continue
        m = (egm - (a * c + b)) ** 2
        if m < best_m:
            best_m, best_o = m, o
    return float(best_o)


LEVEL_FEATURES = ["v50", "v200", "gr_off", "eval_len", "known_range"]


def extract_level_features(well):
    h = well.horizontal
    kn = h[h[TARGET_INPUT].notna()]
    kt = kn[TARGET_INPUT].values
    n = len(kt); w50 = min(50, n - 1); w200 = min(200, n - 1)
    return {
        "v50": (kt[-1] - kt[-1 - w50]) / w50 if w50 > 0 else 0.0,
        "v200": (kt[-1] - kt[-1 - w200]) / w200 if w200 > 0 else 0.0,
        "gr_off": _gr_offset_raw(well),
        "eval_len": int(well.eval_mask.sum()),
        "known_range": float(np.nanmax(kt) - np.nanmin(kt)),
    }


def fit_level_model(train_wells, alpha=1.0):
    X, y = [], []
    for w in train_wells:
        h = w.horizontal
        kn = h[h[TARGET_INPUT].notna()]; evz = h[w.eval_mask]
        if len(evz) < 5 or len(kn) < 200 or "TVT" not in h.columns:
            continue
        f = extract_level_features(w)
        X.append([f[k] for k in LEVEL_FEATURES])
        y.append(float(evz["TVT"].mean()) - float(kn[TARGET_INPUT].iloc[-1]))
    return Ridge(alpha=alpha).fit(np.asarray(X), np.asarray(y))


def level_regression(well, model):
    h = well.horizontal
    kn = h[h[TARGET_INPUT].notna()]
    last = float(kn[TARGET_INPUT].iloc[-1])
    f = extract_level_features(well)
    off = float(model.predict(np.asarray([[f[k] for k in LEVEL_FEATURES]]))[0])
    out = baseline_ffill(h)
    out[well.eval_mask.to_numpy()] = last + off
    return out


# ---- 2nd-order HMM ----------------------------------------------------------
def _hmm_shift(F1, deltas, sig_cells):
    P, R = F1.shape; idx = np.arange(P); out = np.empty_like(F1)
    for r in range(R):
        src = idx - deltas[r]; i0 = np.floor(src).astype(int); frac = src - i0
        i0c = np.clip(i0, 0, P - 1); i1c = np.clip(i0 + 1, 0, P - 1)
        out[:, r] = F1[i0c, r] * (1 - frac) + F1[i1c, r] * frac
    if sig_cells > 0.3:
        out = gaussian_filter1d(out, sig_cells, axis=0, mode="nearest")
    return out


def _hmm_shift_T(G, deltas, sig_cells):
    P, R = G.shape
    if sig_cells > 0.3:
        G = gaussian_filter1d(G, sig_cells, axis=0, mode="nearest")
    idx = np.arange(P); out = np.empty_like(G)
    for r in range(R):
        src = idx + deltas[r]; i0 = np.floor(src).astype(int); frac = src - i0
        i0c = np.clip(i0, 0, P - 1); i1c = np.clip(i0 + 1, 0, P - 1)
        out[:, r] = G[i0c, r] * (1 - frac) + G[i1c, r] * frac
    return out


def _hmm2_fb(gr_obs, md, gr_grid, grid, rates, gs, step,
             mom, sig_r, sig_p, start_p, start_sig, ir, r0_sig, lam, df, emission):
    S = len(gr_obs); P = len(grid); R = len(rates)
    zsc = (gr_obs[:, None] - gr_grid[None, :]) / gs
    if emission == "t":
        em = -0.5 * (df + 1.0) * np.log1p(zsc ** 2 / df) * lam
    else:
        em = -0.5 * np.minimum(zsc ** 2, 600.0) * lam
    Krate = np.exp(-0.5 * ((rates[None, :] - mom * rates[:, None]) / sig_r) ** 2)
    Krate /= Krate.sum(1, keepdims=True)
    dm = np.diff(md, prepend=md[0])
    pos0 = np.exp(-0.5 * ((np.arange(P) - start_p) / (start_sig / step)) ** 2)
    rate0 = np.exp(-0.5 * ((rates - ir) / r0_sig) ** 2)
    if pos0.sum() <= 0 or not np.isfinite(pos0.sum()):
        pos0 = np.ones(P)
    if rate0.sum() <= 0 or not np.isfinite(rate0.sum()):
        rate0 = np.ones(R)
    F = np.outer(pos0, rate0); F /= F.sum()
    a_pred = np.empty((S, P, R)); a_filt = np.empty((S, P, R))
    for s in range(S):
        F2 = F if s == 0 else _hmm_shift(F @ Krate, rates * dm[s] / step, sig_p / step)
        a_pred[s] = F2
        lik = np.exp(em[s] - em[s].max())
        F = F2 * lik[:, None]; F = F / (F.sum() + 1e-300); a_filt[s] = F
    gamma = np.empty((S, P, R)); gamma[S - 1] = a_filt[S - 1]
    for s in range(S - 2, -1, -1):
        ratio = gamma[s + 1] / (a_pred[s + 1] + 1e-300)
        back = _hmm_shift_T(ratio, rates * dm[s + 1] / step, sig_p / step)
        g = a_filt[s] * (back @ Krate.T); gamma[s] = g / (g.sum() + 1e-300)
    post_p = gamma.sum(2); post_p /= post_p.sum(1, keepdims=True)
    mean = post_p @ grid
    var = np.maximum(post_p @ (grid ** 2) - mean ** 2, 0.0)
    return mean, var


def hmm_posterior(well, s_target=250, step=0.5, band_pad=90.0, n_rates=21,
                  rate_span=0.035, mom=0.98, sig_r=0.003, sig_p=0.45, lam=1.0,
                  df=4.0, emission="t"):
    h, t = well.horizontal, well.typewell
    kn = h[h[TARGET_INPUT].notna()]; ev = h[h[TARGET_INPUT].isna()]
    ff = baseline_ffill(h); std_full = np.zeros(len(h))
    if len(ev) == 0 or len(kn) < 50:
        return ff, std_full
    tw_tvt = t["TVT"].to_numpy(float); tw_gr = t["GR"].ffill().bfill().to_numpy(float)
    last = float(kn[TARGET_INPUT].iloc[-1])
    tw_at = np.interp(kn[TARGET_INPUT].values, tw_tvt, tw_gr)
    gs = float(np.clip(np.nanstd(kn["GR"].fillna(np.nanmean(tw_gr)).values - tw_at), 10.0, 60.0))
    tail = kn.tail(30)
    ir = float((tail[TARGET_INPUT].iloc[-1] - tail[TARGET_INPUT].iloc[0]) /
               max(tail["MD"].iloc[-1] - tail["MD"].iloc[0], 1.0))
    gmin = max(tw_tvt.min() - 40, last - band_pad); gmax = min(tw_tvt.max() + 40, last + band_pad)
    grid = np.arange(gmin, gmax + step, step); gr_grid = np.interp(grid, tw_tvt, tw_gr)
    gr_all = h["GR"].interpolate(limit_direction="both").fillna(np.nanmean(tw_gr)).to_numpy(float)
    ei = ev.index.values; gr_e = gr_all[ei]; md_e = ev["MD"].to_numpy(float)
    n = len(ei); S = min(s_target, n); edges = np.linspace(0, n, S + 1).astype(int)
    gr_b = np.array([gr_e[a:b].mean() for a, b in zip(edges[:-1], edges[1:])])
    md_b = np.array([md_e[a:b].mean() for a, b in zip(edges[:-1], edges[1:])])
    rates = np.linspace(-rate_span, rate_span, n_rates); start_p = (last - gmin) / step
    try:
        mean_b, var_b = _hmm2_fb(gr_b, md_b, gr_grid, grid, rates, gs, step,
                                 mom, sig_r, sig_p, start_p, 3.0, ir, 0.01, lam, df, emission)
    except Exception:
        return ff, std_full
    centers = 0.5 * (edges[:-1] + edges[1:])
    pred_e = np.interp(np.arange(n), centers, mean_b)
    std_e = np.sqrt(np.interp(np.arange(n), centers, var_b))
    pred_e = np.where(np.isfinite(pred_e), pred_e, last)
    std_e = np.where(np.isfinite(std_e) & (std_e > 0), std_e, 1e6)
    out = ff.copy(); out[ei] = pred_e; std_full[ei] = std_e
    return out, std_full


def hmm_fusion(well, level_model, blend_k=4.0):
    hp, hs = hmm_posterior(well)
    anchor = level_regression(well, level_model)
    em = well.eval_mask.to_numpy(); out = anchor.copy()
    wt = 1.0 / (1.0 + (hs[em] / blend_k) ** 2)
    out[em] = wt * hp[em] + (1 - wt) * anchor[em]
    return out



# ---- GBM stack --------------------------------------------------------------
from sklearn.ensemble import HistGradientBoostingRegressor

STACK_FEATS = ["hmm_pred", "hmm_std", "z_from_ps", "md_from_ps", "horiz_from_ps", "dz",
               "tort", "gr50", "gr_dev", "az_sin", "az_cos", "eval_frac",
               "v50", "v200", "gr_off", "known_range", "eval_len"]


# ---- particle filter --------------------------------------------------------
def _sysresample(w, rng):
    n = len(w); pos = (rng.random() + np.arange(n)) / n
    return np.clip(np.searchsorted(np.cumsum(w), pos), 0, n - 1)


def pf_posterior(well, n_particles=600, s_target=250, band_pad=90.0,
                 mom=0.98, sig_r=0.003, sig_p=0.45, lam=1.0, df=4.0, seed=0, return_ll=False):
    rng = np.random.default_rng(seed)
    h, t = well.horizontal, well.typewell
    kn = h[h[TARGET_INPUT].notna()]; ev = h[h[TARGET_INPUT].isna()]
    ff = baseline_ffill(h).astype(float); std = np.zeros(len(h))
    if len(ev) == 0 or len(kn) < 50:
        return (ff, std, -1e18) if return_ll else (ff, std)
    tw_tvt = t["TVT"].to_numpy(float)
    tw_gr = gaussian_filter1d(t["GR"].ffill().bfill().to_numpy(float), 3)
    last = float(kn[TARGET_INPUT].iloc[-1])
    tw_at = np.interp(kn[TARGET_INPUT].values, tw_tvt, tw_gr)
    gs = float(np.clip(np.nanstd(kn["GR"].fillna(np.nanmean(tw_gr)).values - tw_at), 10.0, 60.0))
    tail = kn.tail(30)
    ir = float((tail[TARGET_INPUT].iloc[-1] - tail[TARGET_INPUT].iloc[0]) /
               max(tail["MD"].iloc[-1] - tail["MD"].iloc[0], 1.0))
    lo = max(tw_tvt.min() - 40, last - band_pad); hi = min(tw_tvt.max() + 40, last + band_pad)
    gr_all = h["GR"].interpolate(limit_direction="both").fillna(np.nanmean(tw_gr)).to_numpy(float)
    ei = ev.index.values; gr_e = gr_all[ei]; md_e = ev["MD"].to_numpy(float)
    n = len(ei); S = min(s_target, n); edges = np.linspace(0, n, S + 1).astype(int)
    gr_b = np.array([gr_e[a:b].mean() for a, b in zip(edges[:-1], edges[1:])])
    md_b = np.array([md_e[a:b].mean() for a, b in zip(edges[:-1], edges[1:])])
    dm = np.diff(md_b, prepend=float(kn["MD"].iloc[-1]))
    pos = last + rng.normal(0, 3.0, n_particles); rate = ir + rng.normal(0, 0.01, n_particles)
    w = np.full(n_particles, 1.0 / n_particles); mean = np.zeros(S); var = np.zeros(S); loglik = 0.0
    for s in range(S):
        rate = mom * rate + rng.normal(0, sig_r, n_particles)
        pos = np.clip(pos + rate * dm[s] + rng.normal(0, sig_p, n_particles), lo, hi)
        zsc = (gr_b[s] - np.interp(pos, tw_tvt, tw_gr)) / gs
        ll = -0.5 * (df + 1.0) * np.log1p(zsc ** 2 / df) * lam
        m = ll.max(); wnew = w * np.exp(ll - m); sw = wnew.sum()
        loglik += m + np.log(sw + 1e-300)
        w = np.full(n_particles, 1.0 / n_particles) if (sw <= 0 or not np.isfinite(sw)) else wnew / sw
        mean[s] = (w * pos).sum(); var[s] = max((w * pos ** 2).sum() - mean[s] ** 2, 0.0)
        if 1.0 / (w ** 2).sum() < n_particles / 2:
            idx = _sysresample(w, rng); pos = pos[idx]; rate = rate[idx]
            w = np.full(n_particles, 1.0 / n_particles)
    centers = 0.5 * (edges[:-1] + edges[1:])
    pred = np.interp(np.arange(n), centers, mean); sd = np.sqrt(np.interp(np.arange(n), centers, var))
    out = ff.copy(); out[ei] = np.where(np.isfinite(pred), pred, last)
    std[ei] = np.where(np.isfinite(sd) & (sd > 0), sd, 1e6)
    return (out, std, float(loglik)) if return_ll else (out, std)


def pf_lik_ensemble(well, scales=(3.0, 5.0, 8.0, 12.0), n_particles=350, n_seeds=8):
    em = well.eval_mask.to_numpy(); ne = int(em.sum())
    if ne == 0:
        return None
    preds = []; lls = []
    for sd in range(n_seeds):
        out, _, ll = pf_posterior(well, n_particles=n_particles, seed=sd, return_ll=True)
        preds.append(out[em]); lls.append(ll)
    preds = np.stack(preds, 0); lls = np.asarray(lls, float); lls = lls - lls.max()
    rows = []
    for sc in scales:
        wts = np.exp(lls / sc); wts /= wts.sum(); rows.append((wts[:, None] * preds).sum(0))
    rows.append(preds.mean(0))
    return np.stack(rows, 0).astype(np.float32)


def _geology_features(well, hp_em, last):
    # typewell Geology LEVEL features (formation labels -> absolute typewell TVT). Orthogonal to
    # the GR-alignment HMM. Biggest gain in the project: stack CV 13.49->13.00 (notes/11).
    n = len(hp_em); t = well.typewell
    if "Geology" not in t.columns:
        return np.zeros(n), np.zeros(n), np.full(n, last)
    g = t["Geology"].astype(str).str.strip()
    valid = (g.notna() & (g != "") & (g != "nan") & t["TVT"].notna()).to_numpy()
    tw_tvt = t["TVT"].to_numpy(float)[valid]; tw_geo = g.to_numpy()[valid]
    if len(tw_tvt) < 20:
        return np.zeros(n), np.zeros(n), np.full(n, last)
    order = np.argsort(tw_tvt); stvt = tw_tvt[order]; sgeo = tw_geo[order]
    fm_med = {f: float(np.median(tw_tvt[tw_geo == f])) for f in np.unique(tw_geo)}
    def form_med(v):
        i = int(np.clip(np.searchsorted(stvt, v), 0, len(stvt) - 1))
        return fm_med[sgeo[i]]
    last_med = form_med(last)
    hmm_med = np.array([form_med(v) for v in hp_em])
    return np.full(n, last_med - last), hmm_med - last, np.full(n, last_med)


_ANCH_OFFS = [-80, -40, -20, -10, -5, 0, 5, 10, 20, 40, 80]
_PF_OFFS = [-30, -15, -8, -4, -2, 0, 2, 4, 8, 15, 30]


def _rich_cols(h, typewell, ei, hp_em, pp_em, last_tvt):
    # reproduced 7.016 features: multi-offset GR-match grids + finer multi-scale GR (notes/12).
    tw_tvt = typewell["TVT"].to_numpy(float)
    tw_gr = gaussian_filter1d(typewell["GR"].ffill().bfill().to_numpy(float), 3)
    o = np.argsort(tw_tvt); tw_tvt, tw_gr = tw_tvt[o], tw_gr[o]
    gr = h["GR"].interpolate(limit_direction="both").fillna(np.nanmean(tw_gr)).to_numpy(float)
    gr_s = pd.Series(gaussian_filter1d(gr, 3)); hgr = gr_s.to_numpy()[ei]
    cols = []
    for off in _ANCH_OFFS:
        cols.append(hgr - np.interp(last_tvt + off, tw_tvt, tw_gr))
    for off in _PF_OFFS:
        cols.append(hgr - np.interp(pp_em + off, tw_tvt, tw_gr))
        cols.append(hgr - np.interp(hp_em + off, tw_tvt, tw_gr))
    cols.append(hgr - np.interp(last_tvt, tw_tvt, tw_gr))
    for wsz in [5, 21, 51, 101]:
        r = gr_s.rolling(wsz, center=True, min_periods=1)
        cols.append(r.mean().to_numpy()[ei]); cols.append(r.std().fillna(0).to_numpy()[ei])
    for lag in [1, 5, 15, 30]:
        cols.append(gr_s.shift(lag).bfill().to_numpy()[ei]); cols.append(gr_s.shift(-lag).ffill().to_numpy()[ei])
    cols.append(gr_s.diff().fillna(0).to_numpy()[ei]); cols.append(gr_s.diff().diff().fillna(0).to_numpy()[ei])
    cols.append(gr_s.rolling(21, center=True, min_periods=1).max().to_numpy()[ei])
    cols.append(np.sqrt(np.maximum((gr_s ** 2).rolling(21, center=True, min_periods=1).mean(), 0)).to_numpy()[ei])
    return cols


_FORMS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]


def build_spatial_ref(wells):
    """Reference cloud (X,Y)->mean formation surface, tagged by well name. Formation surfaces are
    TRAIN-only but globally consistent, so imputing them at a well's X,Y from neighbour wells is
    test-legal. +0.05 pooled CV, safe (GBM gates on nndist). notes/13."""
    from scipy.spatial import cKDTree
    pts, surf, wid = [], [], []
    for w in wells:
        h = w.horizontal
        if not all(f in h.columns for f in _FORMS):
            continue
        X = h["X"].to_numpy(); Y = h["Y"].to_numpy(); F = h[_FORMS].to_numpy()
        m = np.isfinite(X) & np.isfinite(Y) & np.isfinite(F).all(1)
        idx = np.where(m)[0][::5]
        if len(idx) == 0:
            continue
        pts.append(np.column_stack([X[idx], Y[idx]])); surf.append(F[idx].mean(1))
        wid.append(np.full(len(idx), w.name, dtype=object))
    if not pts:
        return None
    return {"tree": cKDTree(np.vstack(pts)), "surf": np.concatenate(surf), "wid": np.concatenate(wid)}


def _spatial_cols(h, ei, em, self_name, ref):
    """sp_drift (boundary-anchored spatial TVT pred - ffill), sp_nndist (log nearest other-well
    dist), sp_impd (imputed structural drift). Self-excluded (LOWO). Falls back to base when no
    close neighbour. notes/13."""
    n = int(em.sum())
    drift = np.zeros(n, np.float64); nnd = np.full(n, 1e6); impd = np.zeros(n, np.float64)
    if ref is None:
        return drift, np.log1p(nnd), impd
    Ti = h[TARGET_INPUT].to_numpy(); Z = h["Z"].to_numpy()
    X = h["X"].to_numpy(); Y = h["Y"].to_numpy()
    pp = np.isfinite(Ti)
    if not pp.any():
        return drift, np.log1p(nnd), impd
    b = np.where(pp)[0][-1]
    tree, surf, wid = ref["tree"], ref["surf"], ref["wid"]
    qrows = np.append(ei, b)
    qpts = np.column_stack([X[qrows], Y[qrows]])
    dd, ii = tree.query(np.nan_to_num(qpts, nan=1e9), k=40)
    same = wid[ii] == self_name
    dd = np.where(same, np.inf, dd)
    order = np.argsort(dd, axis=1)
    dd = np.take_along_axis(dd, order, 1)[:, :8]
    ii = np.take_along_axis(ii, order, 1)[:, :8]
    wgt = 1.0 / (dd + 1.0); wsum = wgt.sum(1)
    imp = (wgt * surf[ii]).sum(1) / np.where(wsum > 0, wsum, np.nan)
    imp_ev = imp[:n]; imp_bnd = imp[n]
    Db = Ti[b] + Z[b]
    sp_pred = imp_ev + (Db - imp_bnd) - Z[ei]
    ok = np.isfinite(sp_pred)
    drift = np.where(ok, sp_pred - float(Ti[b]), 0.0)
    di = imp_ev - imp_bnd
    impd = np.where(np.isfinite(di), di, 0.0)
    nn_ev = dd[:n, 0]
    nnd = np.where(np.isfinite(nn_ev), nn_ev, 1e6)
    return drift, np.log1p(nnd), impd


def _seg_b_well(ktvt, kz, form_col):
    bv = ktvt + kz - form_col; n = len(bv); b_full = float(np.median(bv))
    b_late = float(np.median(bv[max(0, n - 50):])) if n >= 5 else b_full
    w = np.exp(0.02 * np.arange(n)); w /= w.sum()
    return b_full, b_late, float(np.dot(w, bv))


class _FormationPlaneKNN:
    """Offset-well datum lever (notes/14): per-well median formation surfaces -> local IDW plane fit
    over k nearest OTHER wells -> formation surface at a query X,Y. Test-legal (needs only X,Y)."""
    def __init__(self, wells, k=10):
        from scipy.spatial import cKDTree
        self.k = k; rows = []
        for w in wells:
            h = w.horizontal
            if not all(c in h.columns for c in _FORMS):
                continue
            df = h[["X", "Y"] + _FORMS].dropna()
            if len(df) == 0:
                continue
            row = {"wid": w.name, "x": float(df.X.median()), "y": float(df.Y.median())}
            for c in _FORMS:
                row[f"{c}_m"] = float(df[c].median())
            rows.append(row)
        self.df = pd.DataFrame(rows); self.wmap = {w: i for i, w in enumerate(self.df.wid)}
        xy = self.df[["x", "y"]].to_numpy(); self.scale = np.where(xy.std(0) < 1e-3, 1., xy.std(0))
        self.tree = cKDTree(xy / self.scale); self.xa = self.df.x.to_numpy(); self.ya = self.df.y.to_numpy()
        self.fa = self.df[[f"{c}_m" for c in _FORMS]].to_numpy(np.float64)

    def impute(self, xy_q, self_wid=None):
        k = self.k; q = xy_q / self.scale; nf = min(k + 5, len(self.df))
        dist, idx = self.tree.query(q, k=nf)
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
            coef = np.stack([np.linalg.pinv(A[r]) @ rhs[r] for r in range(len(q))])
        Xq = xy_q[:, 0]; Yq = xy_q[:, 1]
        pred = (Xq[:, None] * coef[:, 0, :] + Yq[:, None] * coef[:, 1, :] + coef[:, 2, :]).astype(np.float32)
        pred[~vk.any(1)] = self.fa.mean(0)
        return pred, np.where(vk, dk, np.inf).min(1).astype(np.float32)


class _DenseANCCImputer:
    def __init__(self, wells, spw=60, k=20):
        from scipy.spatial import cKDTree
        self.k = k; xs, ys, an, wd = [], [], [], []
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

    def impute(self, xy_q, self_wid=None, nfetch=5000):
        k = self.k; xy_q = np.atleast_2d(xy_q); q = xy_q / self.scale; nf = min(nfetch, len(self.ancc))
        dist, idx = self.tree.query(q, k=nf)
        if self_wid:
            dist = np.where(self.wids[idx] == self_wid, np.inf, dist)
        ordr = np.argpartition(dist, min(k - 1, nf - 1), 1)[:, :k]
        dk = np.take_along_axis(dist, ordr, 1); ik = np.take_along_axis(idx, ordr, 1)
        vk = np.isfinite(dk); w = np.where(vk, 1. / (dk + 1e-3), 0.); sw = w.sum(1); safe = np.where(sw < 1e-9, 1., sw)
        a = self.ancc[ik]; ap = (a * w).sum(1) / safe; ap = np.where(sw < 1e-9, float(self.ancc.mean()), ap)
        var = ((a - ap[:, None]) ** 2 * w).sum(1) / safe
        return ap.astype(np.float32), np.sqrt(np.maximum(var, 0.)).astype(np.float32), np.where(vk, dk, np.inf).min(1).astype(np.float32)


def build_formation_ref(wells):
    try:
        FI = _FormationPlaneKNN(wells)
        if len(FI.df) == 0:
            return None
        return {"FI": FI, "DI": _DenseANCCImputer(wells)}
    except Exception:
        return None


def _formation_cols(h, ei, em, self_name, anchor, fref):
    n = int(em.sum())
    if fref is None:
        return [np.zeros(n, np.float32)] * 22
    Ti = h[TARGET_INPUT].to_numpy(); Z = h["Z"].to_numpy(); xy = h[["X", "Y"]].to_numpy()
    eidx = np.where(em)[0]; kidx = np.where(np.isfinite(Ti))[0]
    xy_e = xy[eidx]; xy_k = xy[kidx]; z_e = Z[eidx]; ktvt = Ti[kidx]; z_k = Z[kidx]
    FI, DI = fref["FI"], fref["DI"]
    form_e, knn_d = FI.impute(xy_e, self_wid=self_name); form_k, _ = FI.impute(xy_k, self_wid=self_name)
    tvtF = []; tvtFw = []; tvtF50 = []
    for fi in range(6):
        bf, bl, bw = _seg_b_well(ktvt, z_k, form_k[:, fi])
        tvtF.append((-z_e + form_e[:, fi] + bf - anchor).astype(np.float32))
        tvtFw.append((-z_e + form_e[:, fi] + bw - anchor).astype(np.float32))
        tvtF50.append((-z_e + form_e[:, fi] + bl - anchor).astype(np.float32))
    d_e, dstd_e, d_dist = DI.impute(xy_e, self_wid=self_name); d_k, _, _ = DI.impute(xy_k, self_wid=self_name)
    bfd, _, _ = _seg_b_well(ktvt, z_k, d_k)
    tvt_dense = (-z_e + d_e + bfd - anchor).astype(np.float32)
    return tvtF + tvtFw + tvtF50 + [tvt_dense, np.log1p(knn_d).astype(np.float32),
                                    np.log1p(d_dist).astype(np.float32), dstd_e.astype(np.float32)]


def _stack_features(well, level_model, spatial_ref=None, formation_ref=None):
    h = well.horizontal
    kn = h[h[TARGET_INPUT].notna()]; ev = h[h[TARGET_INPUT].isna()]
    em = well.eval_mask.to_numpy()
    if em.sum() == 0 or len(kn) < 200:
        return None
    kt = kn[TARGET_INPUT].values
    ps_md = float(kn["MD"].iloc[-1]); ps_z = float(kn["Z"].iloc[-1])
    ps_x = float(kn["X"].iloc[-1]); ps_y = float(kn["Y"].iloc[-1])
    hp, hs = hmm_posterior(well)
    ei = ev.index.values
    z = h["Z"].values; md = h["MD"].values; x = h["X"].values; y = h["Y"].values
    dz = np.gradient(z, md)
    tort = pd.Series(dz).rolling(101, center=True, min_periods=1).std().to_numpy()
    gr50 = gaussian_filter1d(h["GR"].interpolate(limit_direction="both").fillna(0).to_numpy(float), 50)
    kn_gr_mean = float(kn["GR"].mean())
    dxe = np.median(np.diff(ev["X"].values)) if len(ev) > 1 else 0.0
    dye = np.median(np.diff(ev["Y"].values)) if len(ev) > 1 else 1.0
    az = np.arctan2(dxe, dye)
    n = len(ei); w50 = min(50, len(kt) - 1); w200 = min(200, len(kt) - 1)
    cols = {
        "hmm_pred": hp[em], "hmm_std": hs[em],
        "z_from_ps": z[ei] - ps_z, "md_from_ps": md[ei] - ps_md,
        "horiz_from_ps": np.sqrt((x[ei] - ps_x) ** 2 + (y[ei] - ps_y) ** 2),
        "dz": dz[ei], "tort": tort[ei], "gr50": gr50[ei], "gr_dev": gr50[ei] - kn_gr_mean,
        "az_sin": np.full(n, np.sin(az)), "az_cos": np.full(n, np.cos(az)),
        "eval_frac": np.arange(n) / n,
        "v50": np.full(n, (kt[-1] - kt[-1 - w50]) / w50),
        "v200": np.full(n, (kt[-1] - kt[-1 - w200]) / w200),
        "gr_off": np.full(n, _gr_offset_raw(well)),
        "known_range": np.full(n, float(np.nanmax(kt) - np.nanmin(kt))),
        "eval_len": np.full(n, n),
    }
    anchor = level_regression(well, level_model)[em]
    wt = 1.0 / (1.0 + (hs[em] / 4.0) ** 2)
    fusion = wt * hp[em] + (1 - wt) * anchor
    pp, ps = pf_posterior(well)
    wtp = 1.0 / (1.0 + (ps[em] / 4.0) ** 2)
    pf_fus = wtp * pp[em] + (1 - wtp) * anchor
    pf_cols = [pp[em], ps[em], pp[em] - anchor, hp[em] - pp[em], pf_fus]
    # Group A: inter-signal uncertainty; Group B: GR envelope/energy on eval rows.
    # Validated on the frontier lockbox: 13.742 -> 13.623 (notes/08).
    hpe, ppe, hse, pse = hp[em], pp[em], hs[em], ps[em]
    a_absdiff = np.abs(hpe - ppe); a_stdsum = hse + pse
    a_cols = [a_absdiff, a_stdsum, a_absdiff / (a_stdsum + 1.0),
              np.abs(hpe - anchor), np.abs(ppe - anchor)]
    tw_gr_b = gaussian_filter1d(well.typewell["GR"].ffill().bfill().to_numpy(float), 3)
    ge = h["GR"].interpolate(limit_direction="both").fillna(np.nanmean(tw_gr_b)).to_numpy(float)[ei]
    gser = pd.Series(ge)
    b_cols = [gser.rolling(51, center=True, min_periods=1).max().to_numpy(),
              gser.rolling(51, center=True, min_periods=1).min().to_numpy(),
              np.sqrt(pd.Series(ge ** 2).rolling(51, center=True, min_periods=1).mean().to_numpy()),
              gser.rolling(31, center=True, min_periods=1).std().fillna(0).to_numpy(),
              gser.rolling(151, center=True, min_periods=1).std().fillna(0).to_numpy(),
              np.abs(np.gradient(gaussian_filter1d(ge, 5))) if len(ge) > 1 else np.zeros(len(ge))]
    # typewell Geology is TRAIN-ONLY (test typewells lack it) -> geology feats were leakage. Removed.
    r_cols = _rich_cols(h, well.typewell, ei, hp[em], pp[em], float(kt[-1]))  # reproduced 7.016 feats (notes/12)
    sp_drift, sp_nndist, sp_impd = _spatial_cols(h, ei, em, well.name, spatial_ref)  # notes/13
    s_cols = [sp_drift, sp_nndist, sp_impd]
    fcols = _formation_cols(h, ei, em, well.name, anchor, formation_ref)  # rich formation plane (notes/14)
    lik = pf_lik_ensemble(well)  # likelihood-weighted multi-seed PF ensemble (notes/14)
    lik_cols = [(lik[i] - anchor).astype(np.float32) for i in range(lik.shape[0])] if lik is not None else [np.zeros(em.sum(), np.float32)] * 5
    X = np.column_stack([cols[k] for k in STACK_FEATS] + [anchor, hp[em] - anchor, fusion, wt]
                        + pf_cols + a_cols + b_cols + r_cols + s_cols + fcols + lik_cols)
    res = dict(X=X, anchor=anchor, fusion=fusion, em=em, mean_dis=float(np.mean(a_absdiff)))
    if "TVT" in h.columns:
        res["truth"] = ev["TVT"].values
    return res


def fit_gbm_stack(train_wells, level_model, step=5, seeds=(0, 1, 2)):
    # Phase 2 + seed-ensemble: LightGBM + CatBoost on every step-th eval row (subsample regularizes;
    # diverse boosters; 3 seeds each averaged removes the seed lottery). Lockbox 13.77 -> 13.466.
    import lightgbm as lgb
    from catboost import CatBoostRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPRegressor
    from joblib import Parallel, delayed
    spatial_ref = build_spatial_ref(train_wells)  # notes/13: impute each well from its neighbours
    formation_ref = build_formation_ref(train_wells)  # notes/14: FormationPlaneKNN + dense ANCC

    def _feat(w):  # per-well feature build (the runtime bottleneck) — threaded over wells
        f = _stack_features(w, level_model, spatial_ref=spatial_ref, formation_ref=formation_ref)
        if f is None or "truth" not in f:
            return None
        return f["X"][::step], (f["truth"] - f["anchor"])[::step], f["mean_dis"]
    parts = [r for r in Parallel(n_jobs=-1, prefer="threads")(delayed(_feat)(w) for w in train_wells) if r is not None]
    X = np.vstack([p[0] for p in parts]); y = np.concatenate([p[1] for p in parts]); dis = [p[2] for p in parts]

    def _fit_lgb(sd):  # GPU LightGBM with CPU fallback (Kaggle build may lack GPU)
        try:
            return lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=63,
                   min_child_samples=200, reg_lambda=1.0, subsample=0.8, subsample_freq=1,
                   colsample_bytree=0.8, random_state=sd, device="gpu", verbose=-1).fit(X, y)
        except Exception:
            return lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=63,
                   min_child_samples=200, reg_lambda=1.0, subsample=0.8, subsample_freq=1,
                   colsample_bytree=0.8, random_state=sd, n_jobs=-1, verbose=-1).fit(X, y)

    def _fit_cat(sd):  # GPU CatBoost (no rsm on GPU) with CPU fallback
        try:
            return CatBoostRegressor(iterations=600, learning_rate=0.03, depth=8, l2_leaf_reg=3.0,
                   random_seed=sd, task_type="GPU", devices="0", verbose=0).fit(X, y)
        except Exception:
            return CatBoostRegressor(iterations=600, learning_rate=0.03, depth=8, l2_leaf_reg=3.0,
                   rsm=0.8, random_seed=sd, thread_count=-1, verbose=0).fit(X, y)
    models = []
    for sd in seeds:
        models.append(_fit_lgb(sd)); models.append(_fit_cat(sd))
    # neural net for diversity (blended 0.7/0.3: -0.20 pooled, notes/13)
    Xn = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    scaler = StandardScaler().fit(Xn)
    tm = np.arange(len(y)) if len(y) <= 200000 else np.random.default_rng(0).choice(len(y), 200000, replace=False)
    Xs = scaler.transform(Xn[tm])
    mlps = [MLPRegressor(hidden_layer_sizes=(256, 128, 64), activation="relu", alpha=1e-3,
                         batch_size=4096, learning_rate_init=1e-3, early_stopping=True,
                         n_iter_no_change=8, max_iter=120, random_state=sd).fit(Xs, y[tm])
            for sd in (0, 1)]
    return {"models": models, "mlps": mlps, "scaler": scaler, "spatial_ref": spatial_ref, "formation_ref": formation_ref,
            "dis_mean": float(np.mean(dis)), "dis_std": float(np.std(dis) + 1e-9)}


def gbm_stack(well, level_model, gbm, blend=0.75):
    f = _stack_features(well, level_model, spatial_ref=gbm.get("spatial_ref"), formation_ref=gbm.get("formation_ref"))
    out = baseline_ffill(well.horizontal).astype(float)
    if f is None:
        return out
    gbm_pred = np.mean([m.predict(f["X"]) for m in gbm["models"]], axis=0)
    if gbm.get("mlps"):
        Xs = gbm["scaler"].transform(np.nan_to_num(f["X"], nan=0.0, posinf=0.0, neginf=0.0))
        pred = 0.7 * gbm_pred + 0.3 * np.mean([m.predict(Xs) for m in gbm["mlps"]], axis=0)
    else:
        pred = gbm_pred
    gbmA = f["anchor"] + pred
    out[f["em"]] = blend * gbmA + (1 - blend) * f["fusion"]
    return out


# ---- run --------------------------------------------------------------------
print("loading train...")
train_wells = load_split(TRAIN_DIR)
print("fitting level model...")
level_model = fit_level_model(train_wells)
print("fitting GBM stack on train (computes HMM per train well)...")
gbm = fit_gbm_stack(train_wells, level_model)

print("predicting test wells...")
from joblib import Parallel, delayed


def _predict(w):
    preds = gbm_stack(w, level_model, gbm); em = w.eval_mask.to_numpy()
    return [{"id": submission_id(w.name, i), "tvt": float(p)} for i, (p, ev) in enumerate(zip(preds, em)) if ev]
rows = [r for part in Parallel(n_jobs=-1, prefer="threads")(delayed(_predict)(w) for w in load_split(TEST_DIR)) for r in part]
sub = pd.DataFrame(rows)
sub.to_csv("submission.csv", index=False)
print(f"wrote submission.csv: {len(sub)} rows, NaN={sub['tvt'].isna().sum()}")
print(sub.head())
