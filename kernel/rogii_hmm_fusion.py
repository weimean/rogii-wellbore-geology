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
                 mom=0.98, sig_r=0.003, sig_p=0.45, lam=1.0, df=4.0, seed=0):
    rng = np.random.default_rng(seed)
    h, t = well.horizontal, well.typewell
    kn = h[h[TARGET_INPUT].notna()]; ev = h[h[TARGET_INPUT].isna()]
    ff = baseline_ffill(h).astype(float); std = np.zeros(len(h))
    if len(ev) == 0 or len(kn) < 50:
        return ff, std
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
    w = np.full(n_particles, 1.0 / n_particles); mean = np.zeros(S); var = np.zeros(S)
    for s in range(S):
        rate = mom * rate + rng.normal(0, sig_r, n_particles)
        pos = np.clip(pos + rate * dm[s] + rng.normal(0, sig_p, n_particles), lo, hi)
        zsc = (gr_b[s] - np.interp(pos, tw_tvt, tw_gr)) / gs
        ll = -0.5 * (df + 1.0) * np.log1p(zsc ** 2 / df) * lam
        w = w * np.exp(ll - ll.max()); sw = w.sum()
        w = np.full(n_particles, 1.0 / n_particles) if (sw <= 0 or not np.isfinite(sw)) else w / sw
        mean[s] = (w * pos).sum(); var[s] = max((w * pos ** 2).sum() - mean[s] ** 2, 0.0)
        if 1.0 / (w ** 2).sum() < n_particles / 2:
            idx = _sysresample(w, rng); pos = pos[idx]; rate = rate[idx]
            w = np.full(n_particles, 1.0 / n_particles)
    centers = 0.5 * (edges[:-1] + edges[1:])
    pred = np.interp(np.arange(n), centers, mean); sd = np.sqrt(np.interp(np.arange(n), centers, var))
    out = ff.copy(); out[ei] = np.where(np.isfinite(pred), pred, last)
    std[ei] = np.where(np.isfinite(sd) & (sd > 0), sd, 1e6)
    return out, std


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


def _stack_features(well, level_model):
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
    X = np.column_stack([cols[k] for k in STACK_FEATS] + [anchor, hp[em] - anchor, fusion, wt]
                        + pf_cols + a_cols + b_cols)
    res = dict(X=X, anchor=anchor, fusion=fusion, em=em)
    if "TVT" in h.columns:
        res["truth"] = ev["TVT"].values
    return res


def fit_gbm_stack(train_wells, level_model, step=5):
    # Phase 2: LightGBM + CatBoost on every step-th eval row (subsample regularizes; two
    # diverse boosters average). Lockbox 13.77 -> 13.505 vs the old single HistGBM.
    import lightgbm as lgb
    from catboost import CatBoostRegressor
    X, y = [], []
    for w in train_wells:
        f = _stack_features(w, level_model)
        if f is None or "truth" not in f:
            continue
        X.append(f["X"][::step]); y.append((f["truth"] - f["anchor"])[::step])
    X = np.vstack(X); y = np.concatenate(y)
    lgm = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=63,
                            min_child_samples=200, reg_lambda=1.0, subsample=0.8, subsample_freq=1,
                            colsample_bytree=0.8, random_state=0, n_jobs=-1, verbose=-1).fit(X, y)
    cbm = CatBoostRegressor(iterations=600, learning_rate=0.03, depth=8, l2_leaf_reg=3.0,
                            rsm=0.8, random_seed=0, thread_count=-1, verbose=0).fit(X, y)
    return {"lgb": lgm, "cat": cbm}


def gbm_stack(well, level_model, gbm, blend=0.6):
    f = _stack_features(well, level_model)
    out = baseline_ffill(well.horizontal).astype(float)
    if f is None:
        return out
    pred = 0.5 * (gbm["lgb"].predict(f["X"]) + gbm["cat"].predict(f["X"]))
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
rows = []
for w in load_split(TEST_DIR):
    preds = gbm_stack(w, level_model, gbm)
    em = w.eval_mask.to_numpy()
    for i, (p, is_ev) in enumerate(zip(preds, em)):
        if is_ev:
            rows.append({"id": submission_id(w.name, i), "tvt": float(p)})
sub = pd.DataFrame(rows)
sub.to_csv("submission.csv", index=False)
print(f"wrote submission.csv: {len(sub)} rows, NaN={sub['tvt'].isna().sum()}")
print(sub.head())
