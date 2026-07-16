"""Models.

Start with a trivial, honest baseline so the whole pipeline (load → feature → predict →
submit → score) runs end to end. Beat it deliberately, one idea at a time, logging each
attempt in experiments/log.md and the vault note.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

from .data import TARGET_INPUT, Well


def baseline_ffill(horizontal: pd.DataFrame, target_input_col: str = TARGET_INPUT) -> np.ndarray:
    return horizontal[target_input_col].ffill().bfill().to_numpy()


def baseline_linear_interp(horizontal: pd.DataFrame, target_input_col: str = TARGET_INPUT) -> np.ndarray:
    s = horizontal.set_index("MD")[target_input_col].interpolate(method="index")
    return s.bfill().ffill().to_numpy()


def kalman_gr(
    well: Well,
    process_std: float = 0.1,
    obs_std: float = 10.0,
    gr_sigma: float = 10.0,
    blend_decay: float = 10_000.0,
    bias: float = 1.42,
) -> np.ndarray:
    """Extended Kalman filter tracking TVT via typewell GR matching.

    Process model: TVT random walk (small steps).
    Observation model: horizontal GR should match typewell GR at current TVT estimate.
    Blends toward ffill at large distances from the PS boundary to limit drift.
    """
    h = well.horizontal
    t = well.typewell
    known = h[h[TARGET_INPUT].notna()]
    evz = h[h[TARGET_INPUT].isna()]

    tw_tvt = t["TVT"].values
    tw_gr_smooth = gaussian_filter1d(t["GR"].values, sigma=3)
    tw_gr_fn = interp1d(tw_tvt, tw_gr_smooth, kind="linear", bounds_error=False, fill_value=np.nan)
    tw_dgr = np.gradient(tw_gr_smooth, 0.5)
    tw_dgr_fn = interp1d(tw_tvt, tw_dgr, kind="linear", bounds_error=False, fill_value=0.0)

    gr_all = h["GR"].values.copy()
    valid = ~np.isnan(gr_all)
    if valid.sum() > 10:
        gr_interp = np.interp(np.arange(len(gr_all)), np.where(valid)[0], gr_all[valid])
        gr_smooth = gaussian_filter1d(gr_interp, sigma=gr_sigma)
    else:
        gr_smooth = np.full(len(gr_all), np.nan)

    last_tvt = float(known[TARGET_INPUT].iloc[-1])
    ps_md = float(known["MD"].iloc[-1])
    preds = h[TARGET_INPUT].ffill().bfill().to_numpy().astype(float)

    mu = last_tvt
    var = 1.0
    Q = process_std**2
    R = obs_std**2

    kalman_preds = np.full(len(h), last_tvt)
    for idx in evz.index:
        mu_p = mu
        var_p = var + Q

        gr_obs = gr_smooth[idx]
        if not np.isnan(gr_obs):
            gr_exp = tw_gr_fn(mu_p)
            if not np.isnan(gr_exp):
                H = float(tw_dgr_fn(mu_p))
                if abs(H) > 0.1:
                    inn = gr_obs - gr_exp
                    S = H**2 * var_p + R
                    K = var_p * H / S
                    mu = mu_p + K * inn
                    var = (1 - K * H) * var_p
                else:
                    mu, var = mu_p, var_p
            else:
                mu, var = mu_p, var_p
        else:
            mu, var = mu_p, var_p

        kalman_preds[idx] = mu

    for idx in evz.index:
        dist = h.loc[idx, "MD"] - ps_md
        w = np.exp(-dist / blend_decay)
        preds[idx] = w * kalman_preds[idx] + (1 - w) * last_tvt + bias

    return preds


def _calibrate_gr(known: pd.DataFrame, tw_gr_fn) -> tuple[float, float]:
    """Learn horizontal_GR ≈ a * typewell_GR + b from known zone."""
    known_with_gr = known.dropna(subset=["GR", TARGET_INPUT])
    if len(known_with_gr) < 50:
        return 1.0, 0.0
    tw_at = tw_gr_fn(known_with_gr[TARGET_INPUT].values)
    valid = ~np.isnan(tw_at)
    if valid.sum() < 20:
        return 1.0, 0.0
    return tuple(np.polyfit(tw_at[valid], known_with_gr["GR"].values[valid], 1))


def _smooth_gr(h: pd.DataFrame, sigma: float) -> np.ndarray:
    """Interpolate GR NaNs and Gaussian-smooth."""
    gr = h["GR"].values.copy()
    valid = ~np.isnan(gr)
    if valid.sum() < 20:
        return np.full(len(gr), np.nan)
    gr_interp = np.interp(np.arange(len(gr)), np.where(valid)[0], gr[valid])
    return gaussian_filter1d(gr_interp, sigma=sigma)


def gr_offset_search(
    well: Well,
    n_segments: int = 1,
    gr_sigma: float = 100.0,
    alpha: float = 0.16,
    bias: float = 1.42,
    max_offset: float = 30.0,
) -> np.ndarray:
    """Find per-segment TVT offset by matching mean GR against calibrated typewell.

    Divides the eval zone into segments, finds the typewell TVT whose calibrated GR
    best matches each segment's mean GR, applies a damped offset (alpha), and
    interpolates smoothly across the eval zone.
    """
    h = well.horizontal
    t = well.typewell
    known = h[h[TARGET_INPUT].notna()]
    evz = h[h[TARGET_INPUT].isna()]
    last_tvt = float(known[TARGET_INPUT].iloc[-1])

    tw_tvt = t["TVT"].values
    tw_gr_smooth = gaussian_filter1d(t["GR"].values, sigma=3)
    tw_gr_fn = interp1d(tw_tvt, tw_gr_smooth, kind="linear", bounds_error=False, fill_value=np.nan)

    a_cal, b_cal = _calibrate_gr(known, tw_gr_fn)

    gr_smooth = _smooth_gr(h, gr_sigma)
    if np.all(np.isnan(gr_smooth)):
        return baseline_ffill(h)

    eval_idx = evz.index.values
    n_eval = len(eval_idx)
    seg_size = max(1, n_eval // n_segments)

    offsets = np.linspace(-max_offset, max_offset, 121)
    seg_results = []
    seg_centers = []

    for i in range(n_segments):
        start = i * seg_size
        end = min((i + 1) * seg_size, n_eval) if i < n_segments - 1 else n_eval
        seg_idx = eval_idx[start:end]
        seg_gr_mean = gr_smooth[seg_idx].mean()
        seg_centers.append(float(h.loc[seg_idx[len(seg_idx) // 2], "MD"]))

        best_mse, best_off = np.inf, 0.0
        for off in offsets:
            tw_gr_cand = tw_gr_fn(last_tvt + off)
            if np.isnan(tw_gr_cand):
                continue
            mse = (seg_gr_mean - (a_cal * tw_gr_cand + b_cal)) ** 2
            if mse < best_mse:
                best_mse = mse
                best_off = off
        seg_results.append(alpha * best_off)

    eval_md = evz["MD"].values
    offset_fn = interp1d(
        seg_centers, seg_results, kind="linear",
        bounds_error=False, fill_value=(seg_results[0], seg_results[-1]),
    )

    preds = baseline_ffill(h).copy().astype(float)
    preds[eval_idx] = last_tvt + offset_fn(eval_md) + bias
    return preds


def ensemble_kalman_offset(
    well: Well,
    w_kalman: float = 0.25,
    w_offset: float = 0.75,
) -> np.ndarray:
    """Weighted blend of Kalman GR tracker and piecewise GR offset search."""
    p_kalman = kalman_gr(well)
    p_offset = gr_offset_search(well)
    return w_kalman * p_kalman + w_offset * p_offset


# --- Level regression --------------------------------------------------------
# EDA (notes/approaches/03) showed the eval-zone error is 68% LEVEL (which constant
# depth), 15% trend, 18% wiggle. Predicting the per-well level offset directly beats
# GR-wiggle matching. The offset (eval_mean - last_known) responds to pre-PS TVT
# velocity + a GR-match signal; a Ridge model recovers ~10% of its variance.

LEVEL_FEATURES = ["v50", "v200", "gr_off", "eval_len", "known_range"]


def _gr_offset_raw(well: Well, gr_sigma: float = 100.0, max_offset: float = 30.0) -> float:
    """Raw best TVT offset from calibrated mean-GR typewell matching (no damping)."""
    h = well.horizontal
    t = well.typewell
    known = h[h[TARGET_INPUT].notna()]
    evz = h[h[TARGET_INPUT].isna()]
    if len(known) == 0 or len(evz) == 0:
        return 0.0
    last_tvt = float(known[TARGET_INPUT].iloc[-1])

    tw_tvt = t["TVT"].values
    tw_gr_smooth = gaussian_filter1d(t["GR"].values, sigma=3)
    tw_gr_fn = interp1d(tw_tvt, tw_gr_smooth, kind="linear", bounds_error=False, fill_value=np.nan)
    a_cal, b_cal = _calibrate_gr(known, tw_gr_fn)

    gr_smooth = _smooth_gr(h, gr_sigma)
    if np.all(np.isnan(gr_smooth)):
        return 0.0
    eval_gr_mean = gr_smooth[evz.index.values].mean()

    best_mse, best_off = np.inf, 0.0
    for off in np.linspace(-max_offset, max_offset, 121):
        tw_gr_cand = tw_gr_fn(last_tvt + off)
        if np.isnan(tw_gr_cand):
            continue
        mse = (eval_gr_mean - (a_cal * tw_gr_cand + b_cal)) ** 2
        if mse < best_mse:
            best_mse, best_off = mse, off
    return float(best_off)


def extract_level_features(well: Well) -> dict:
    """Per-well features for the level offset. Uses only known zone + typewell,
    so it is identical between train and test (no leakage)."""
    h = well.horizontal
    known = h[h[TARGET_INPUT].notna()]
    kt = known[TARGET_INPUT].values
    n = len(kt)
    w50 = min(50, n - 1)
    w200 = min(200, n - 1)
    return {
        "v50": (kt[-1] - kt[-1 - w50]) / w50 if w50 > 0 else 0.0,
        "v200": (kt[-1] - kt[-1 - w200]) / w200 if w200 > 0 else 0.0,
        "gr_off": _gr_offset_raw(well),
        "eval_len": int(well.eval_mask.sum()),
        "known_range": float(np.nanmax(kt) - np.nanmin(kt)),
    }


def fit_level_model(train_wells: list[Well], alpha: float = 1.0):
    """Fit a Ridge model predicting the eval-zone offset (eval_mean - last_known)."""
    from sklearn.linear_model import Ridge

    X, y = [], []
    for w in train_wells:
        h = w.horizontal
        known = h[h[TARGET_INPUT].notna()]
        evz = h[w.eval_mask]
        if len(evz) < 5 or len(known) < 200:
            continue
        feats = extract_level_features(w)
        X.append([feats[k] for k in LEVEL_FEATURES])
        y.append(float(evz["TVT"].mean()) - float(known[TARGET_INPUT].iloc[-1]))
    return Ridge(alpha=alpha).fit(np.asarray(X), np.asarray(y))


def level_regression(well: Well, model, shape_weight: float = 0.0) -> np.ndarray:
    """Predict a constant-per-well level (last_known + Ridge offset), optionally
    adding a fraction of the Kalman within-zone shape."""
    h = well.horizontal
    known = h[h[TARGET_INPUT].notna()]
    last_tvt = float(known[TARGET_INPUT].iloc[-1])

    feats = extract_level_features(well)
    x = np.asarray([[feats[k] for k in LEVEL_FEATURES]])
    offset = float(model.predict(x)[0])

    preds = baseline_ffill(h).astype(float)
    em = well.eval_mask.to_numpy()
    preds[em] = last_tvt + offset

    if shape_weight > 0:
        kal = kalman_gr(well)
        preds[em] += shape_weight * (kal[em] - kal[em].mean())
    return preds


# --- HMM fusion --------------------------------------------------------------
# Our own 2nd-order (TVT, dip-rate) forward-backward HMM smoother, informed by (not
# copied from) the winning notebooks. Joint state = (position, dip-rate) with an OU
# mean-reverting rate (matches the bounded-drift physics from our variance-vs-lag EDA);
# a grid HMM represents multi-modal posteriors so the smoother survives GR aliasing.
# Fused with level_regression by posterior variance: trust the HMM where GR is
# informative (low var), fall back to the level anchor where it isn't. See notes/05.

def _hmm_shift(F1: np.ndarray, deltas: np.ndarray, sig_cells: float) -> np.ndarray:
    """Shift each rate-column of F1 (P,R) by deltas[r] cells (linear interp) then smooth."""
    P, R = F1.shape
    idx = np.arange(P)
    out = np.empty_like(F1)
    for r in range(R):
        src = idx - deltas[r]
        i0 = np.floor(src).astype(int); frac = src - i0
        i0c = np.clip(i0, 0, P - 1); i1c = np.clip(i0 + 1, 0, P - 1)
        out[:, r] = F1[i0c, r] * (1 - frac) + F1[i1c, r] * frac
    if sig_cells > 0.3:
        out = gaussian_filter1d(out, sig_cells, axis=0, mode="nearest")
    return out


def _hmm_shift_T(G: np.ndarray, deltas: np.ndarray, sig_cells: float) -> np.ndarray:
    """Transpose of _hmm_shift (Gaussian ~symmetric; shift transpose = shift by -delta)."""
    P, R = G.shape
    if sig_cells > 0.3:
        G = gaussian_filter1d(G, sig_cells, axis=0, mode="nearest")
    idx = np.arange(P)
    out = np.empty_like(G)
    for r in range(R):
        src = idx + deltas[r]
        i0 = np.floor(src).astype(int); frac = src - i0
        i0c = np.clip(i0, 0, P - 1); i1c = np.clip(i0 + 1, 0, P - 1)
        out[:, r] = G[i0c, r] * (1 - frac) + G[i1c, r] * frac
    return out


def _hmm2_fb(gr_obs, md, gr_grid, grid, rates, gs, step,
             mom, sig_r, sig_p, start_p, start_sig, ir, r0_sig, lam, df, emission):
    """2nd-order (position, dip-rate) forward-backward HMM. Returns (mean, var) per obs."""
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
        if s == 0:
            F2 = F
        else:
            F1 = F @ Krate
            F2 = _hmm_shift(F1, rates * dm[s] / step, sig_p / step)
        a_pred[s] = F2
        lik = np.exp(em[s] - em[s].max())
        F = F2 * lik[:, None]
        F = F / (F.sum() + 1e-300)
        a_filt[s] = F

    gamma = np.empty((S, P, R))
    gamma[S - 1] = a_filt[S - 1]
    for s in range(S - 2, -1, -1):
        ratio = gamma[s + 1] / (a_pred[s + 1] + 1e-300)
        back = _hmm_shift_T(ratio, rates * dm[s + 1] / step, sig_p / step)
        g = a_filt[s] * (back @ Krate.T)
        gamma[s] = g / (g.sum() + 1e-300)

    post_p = gamma.sum(2)
    post_p /= post_p.sum(1, keepdims=True)
    mean = post_p @ grid
    var = np.maximum(post_p @ (grid ** 2) - mean ** 2, 0.0)
    return mean, var


def hmm_posterior(well: Well, s_target: int = 250, step: float = 0.5, band_pad: float = 90.0,
                  n_rates: int = 21, rate_span: float = 0.035, mom: float = 0.98,
                  sig_r: float = 0.003, sig_p: float = 0.45, lam: float = 1.0,
                  df: float = 4.0, emission: str = "t"):
    """Run the 2nd-order HMM on a well; return (pred_rows, std_rows) over ALL rows
    (known rows = ffill, eval rows = posterior mean/std). Falls back to ffill on failure."""
    h, t = well.horizontal, well.typewell
    kn = h[h[TARGET_INPUT].notna()]; ev = h[h[TARGET_INPUT].isna()]
    ff = baseline_ffill(h).astype(float)
    std_full = np.zeros(len(h))
    if len(ev) == 0 or len(kn) < 50:
        return ff, std_full

    tw_tvt = t["TVT"].to_numpy(float)
    tw_gr = t["GR"].ffill().bfill().to_numpy(float)
    last_tvt = float(kn[TARGET_INPUT].iloc[-1])

    tw_at = np.interp(kn[TARGET_INPUT].values, tw_tvt, tw_gr)
    gs = float(np.clip(np.nanstd(kn["GR"].fillna(np.nanmean(tw_gr)).values - tw_at), 10.0, 60.0))
    tail = kn.tail(30)
    ir = float((tail[TARGET_INPUT].iloc[-1] - tail[TARGET_INPUT].iloc[0]) /
               max(tail["MD"].iloc[-1] - tail["MD"].iloc[0], 1.0))

    gmin = max(tw_tvt.min() - 40, last_tvt - band_pad)
    gmax = min(tw_tvt.max() + 40, last_tvt + band_pad)
    grid = np.arange(gmin, gmax + step, step)
    gr_grid = np.interp(grid, tw_tvt, tw_gr)

    gr_all = h["GR"].interpolate(limit_direction="both").fillna(np.nanmean(tw_gr)).to_numpy(float)
    ei = ev.index.values
    gr_e = gr_all[ei]; md_e = ev["MD"].to_numpy(float)
    n = len(ei); S = min(s_target, n)
    edges = np.linspace(0, n, S + 1).astype(int)
    gr_b = np.array([gr_e[a:b].mean() for a, b in zip(edges[:-1], edges[1:])])
    md_b = np.array([md_e[a:b].mean() for a, b in zip(edges[:-1], edges[1:])])

    rates = np.linspace(-rate_span, rate_span, n_rates)
    start_p = (last_tvt - gmin) / step
    try:
        mean_b, var_b = _hmm2_fb(gr_b, md_b, gr_grid, grid, rates, gs, step,
                                 mom, sig_r, sig_p, start_p, 3.0, ir, 0.01, lam, df, emission)
    except Exception:
        return ff, std_full

    centers = 0.5 * (edges[:-1] + edges[1:])
    pred_e = np.interp(np.arange(n), centers, mean_b)
    std_e = np.sqrt(np.interp(np.arange(n), centers, var_b))
    pred_e = np.where(np.isfinite(pred_e), pred_e, last_tvt)
    std_e = np.where(np.isfinite(std_e) & (std_e > 0), std_e, 1e6)

    out = ff.copy(); out[ei] = pred_e
    std_full[ei] = std_e
    return out, std_full


def hmm_fusion(well: Well, level_model, blend_k: float = 4.0, **hmm_kw) -> np.ndarray:
    """Fuse the HMM posterior with the level_regression anchor by posterior variance:
    w_hmm = 1/(1+(std/blend_k)^2). Our best model (CV RMSE ~14.24, -10.5% vs ffill)."""
    hmm_pred, hmm_std = hmm_posterior(well, **hmm_kw)
    anchor = level_regression(well, level_model)
    em = well.eval_mask.to_numpy()
    out = anchor.copy()
    wt = 1.0 / (1.0 + (hmm_std[em] / blend_k) ** 2)
    out[em] = wt * hmm_pred[em] + (1 - wt) * anchor[em]
    return out


# --- GBM stack ---------------------------------------------------------------
# Stacks the HMM posterior + level anchor + trajectory features in a gradient-boosted
# tree that learns a residual correction to the level anchor; the final prediction blends
# that correction 50/50 with the hand-tuned variance fusion (they are complementary).
# Full 773-well GroupKFold CV: 13.77 (vs hmm_fusion 14.24, ffill 15.91).

STACK_FEATS = ["hmm_pred", "hmm_std", "z_from_ps", "md_from_ps", "horiz_from_ps", "dz",
               "tort", "gr50", "gr_dev", "az_sin", "az_cos", "eval_frac",
               "v50", "v200", "gr_off", "known_range", "eval_len"]


def _geology_features(well: Well, hp_em: np.ndarray, last: float):
    """Typewell-Geology LEVEL features (the one geological signal available at test time).
    A lateral holds inside a target formation whose absolute TVT the typewell gives us:
      geo_last_off  = median TVT of last-known's formation - last  (test-legal offset cue)
      geo_hmm_off   = median TVT of the HMM-aligned eval formation - last  (GR-driven)
      geo_last_med  = that formation's median TVT  (absolute level reference)
    Orthogonal to the HMM (GR *alignment*, not formation *labels*): stack CV 13.49->13.00."""
    n = len(hp_em)
    t = well.typewell
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
    """Reproduced 7.016 features (test-legal): multi-offset GR-match grids (a GR-match profile
    around last_tvt/PF/HMM) + finer multi-scale GR roll/lag/lead. CV 13.47->13.20 (notes/12)."""
    import pandas as pd
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
    """Reference cloud of (X,Y) -> mean formation surface, tagged by well name, for spatial
    imputation. Formation surfaces (ANCC..BUDA) are TRAIN-only but globally consistent absolute
    depths, so imputing them at a well's X,Y from NEIGHBOUR wells is test-legal (never touches
    the well's own — absent-at-test — formation columns). notes/13."""
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
    pts = np.vstack(pts); surf = np.concatenate(surf); wid = np.concatenate(wid)
    return {"tree": cKDTree(pts), "surf": surf, "wid": wid}


def _spatial_cols(h, ei, em, self_name, ref):
    """Spatial-formation features for eval rows: sp_drift (boundary-anchored spatial TVT
    prediction - ffill), sp_nndist (log nearest OTHER-well distance = trust gate), sp_impd
    (imputed structural drift). Excludes self_name from the cloud (LOWO at fit; a test well is
    simply absent). Mirrors _cache_spatial.py. +0.05 pooled CV, safe (GBM gates on nndist)."""
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
    last_tvt = float(Ti[b])
    drift = np.where(ok, sp_pred - last_tvt, 0.0)
    di = imp_ev - imp_bnd
    impd = np.where(np.isfinite(di), di, 0.0)
    nn_ev = dd[:n, 0]
    nnd = np.where(np.isfinite(nn_ev), nn_ev, 1e6)
    return drift, np.log1p(nnd), impd


def _seg_b_well(ktvt, kz, form_col):
    """Per-well datum offset b = median(TVT_known + Z_known − imputed_formation), in segment variants
    (full / last-50 / exp-weighted-tail). notes/14."""
    bv = ktvt + kz - form_col; n = len(bv); b_full = float(np.median(bv))
    b_late = float(np.median(bv[max(0, n - 50):])) if n >= 5 else b_full
    w = np.exp(0.02 * np.arange(n)); w /= w.sum()
    return b_full, b_late, float(np.dot(w, bv))


class _FormationPlaneKNN:
    """Top-team offset-well datum lever (notes/14): per train well store (median X,Y, median depth of
    each of 6 formations); for a query point fit a local IDW weighted PLANE depth~a·X+b·Y+c over the
    k nearest OTHER wells (self-excluded) → formation surface at the query X,Y. Needs only X,Y at
    inference — the test well's own (absent) formation columns are never read."""
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
        import pandas as _pd
        self.df = _pd.DataFrame(rows); self.wmap = {w: i for i, w in enumerate(self.df.wid)}
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
    """Dense IDW-KNN over all wells' subsampled (X,Y,ANCC) points (self-excluded). notes/14."""
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
    """Build the FormationPlaneKNN + DenseANCC imputers from train wells (they carry the TRAIN-only
    formation columns). Returns None if no well has formations (e.g. a test-only call)."""
    try:
        FI = _FormationPlaneKNN(wells)
        if len(FI.df) == 0:
            return None
        return {"FI": FI, "DI": _DenseANCCImputer(wells)}
    except Exception:
        return None


def _formation_cols(h, ei, em, self_name, anchor, fref):
    """Rich spatial formation features (notes/14): tvtF_{6} (b_full), tvtFw_{6} (b_wls), tvtF50_{6}
    (b_late), tvt_dense — each as a CORRECTION to the anchor (≈ y where a close neighbour exists) —
    plus plane_knn_d / dense_dist / dense_std gates. 22 columns. Returns zeros if fref is None."""
    n = int(em.sum())
    if fref is None:
        return [np.zeros(n, np.float32)] * 22
    Ti = h[TARGET_INPUT].to_numpy(); Z = h["Z"].to_numpy()
    xy = h[["X", "Y"]].to_numpy()
    eidx = np.where(em)[0]; kidx = np.where(np.isfinite(Ti))[0]
    xy_e = xy[eidx]; xy_k = xy[kidx]; z_e = Z[eidx]; ktvt = Ti[kidx]; z_k = Z[kidx]
    FI, DI = fref["FI"], fref["DI"]
    form_e, knn_d = FI.impute(xy_e, self_wid=self_name)
    form_k, _ = FI.impute(xy_k, self_wid=self_name)
    tvtF = []; tvtFw = []; tvtF50 = []
    for fi in range(6):
        bf, bl, bw = _seg_b_well(ktvt, z_k, form_k[:, fi])
        tvtF.append((-z_e + form_e[:, fi] + bf - anchor).astype(np.float32))
        tvtFw.append((-z_e + form_e[:, fi] + bw - anchor).astype(np.float32))
        tvtF50.append((-z_e + form_e[:, fi] + bl - anchor).astype(np.float32))
    d_e, dstd_e, d_dist = DI.impute(xy_e, self_wid=self_name)
    d_k, _, _ = DI.impute(xy_k, self_wid=self_name)
    bfd, _, _ = _seg_b_well(ktvt, z_k, d_k)
    tvt_dense = (-z_e + d_e + bfd - anchor).astype(np.float32)
    return tvtF + tvtFw + tvtF50 + [tvt_dense, np.log1p(knn_d).astype(np.float32),
                                    np.log1p(d_dist).astype(np.float32), dstd_e.astype(np.float32)]


def _stack_features(well: Well, level_model, anchor_override=None, spatial_ref=None, formation_ref=None):
    """Per-eval-row feature matrix + anchor/fusion/ff arrays for the GBM stack.
    Returns None if the well is too short. Includes 'truth' only when TVT is present.
    ``anchor_override`` (eval-row array) substitutes an out-of-fold level anchor for
    honest CV; when None the in-sample ``level_regression`` prediction is used."""
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
    anchor = anchor_override if anchor_override is not None else level_regression(well, level_model)[em]
    wt = 1.0 / (1.0 + (hs[em] / 4.0) ** 2)
    fusion = wt * hp[em] + (1 - wt) * anchor
    # particle filter — a diverse tracker; its columns add ~0.24 RMSE to the stack
    pp, ps = pf_posterior(well)
    wtp = 1.0 / (1.0 + (ps[em] / 4.0) ** 2)
    pf_fus = wtp * pp[em] + (1 - wtp) * anchor
    pf_cols = [pp[em], ps[em], pp[em] - anchor, hp[em] - pp[em], pf_fus]
    # --- Group A: inter-signal uncertainty (HMM vs PF disagreement / anchor distance).
    #     Validated on the frontier lockbox with Group B: 13.742 -> 13.623 (notes/08). ---
    hpe, ppe, hse, pse = hp[em], pp[em], hs[em], ps[em]
    a_absdiff = np.abs(hpe - ppe)
    a_stdsum = hse + pse
    a_cols = [a_absdiff, a_stdsum, a_absdiff / (a_stdsum + 1.0),
              np.abs(hpe - anchor), np.abs(ppe - anchor)]
    # --- Group B: GR envelope/energy on eval rows (rolling max/min/rms/std + |dGR|).
    #     Cleanest lockbox win: 13.742 -> 13.650 at ~no dev-CV cost. ---
    tw_gr_b = gaussian_filter1d(well.typewell["GR"].ffill().bfill().to_numpy(float), 3)
    ge = h["GR"].interpolate(limit_direction="both").fillna(np.nanmean(tw_gr_b)).to_numpy(float)[ei]
    gser = pd.Series(ge)
    b_cols = [gser.rolling(51, center=True, min_periods=1).max().to_numpy(),
              gser.rolling(51, center=True, min_periods=1).min().to_numpy(),
              np.sqrt(pd.Series(ge ** 2).rolling(51, center=True, min_periods=1).mean().to_numpy()),
              gser.rolling(31, center=True, min_periods=1).std().fillna(0).to_numpy(),
              gser.rolling(151, center=True, min_periods=1).std().fillna(0).to_numpy(),
              np.abs(np.gradient(gaussian_filter1d(ge, 5))) if len(ge) > 1 else np.zeros(len(ge))]
    # NOTE: typewell Geology is TRAIN-ONLY (test typewells have only TVT, GR). Geology LEVEL
    # features looked like a -0.53 CV win but were leakage — absent at inference (notes/11). Removed.
    # --- Reproduced 7.016 rich features (notes/12): multi-offset GR-match grids + finer GR.
    #     Biggest single gain since the HMM: pooled CV 13.47->13.20, lockbox 13.30->13.00. ---
    r_cols = _rich_cols(h, well.typewell, ei, hp[em], pp[em], float(kt[-1]))
    # --- Spatial formation features (notes/13): impute the TRAIN-only structural surfaces at this
    #     well's X,Y from NEIGHBOUR wells; +0.05 pooled CV. Safe — GBM gates on sp_nndist, so a
    #     well with no close train neighbour falls back to the base prediction (can't hurt). ---
    sp_drift, sp_nndist, sp_impd = _spatial_cols(h, ei, em, well.name, spatial_ref)
    s_cols = [sp_drift, sp_nndist, sp_impd]
    # --- Rich spatial formation-PLANE features (notes/14): the top-team datum lever — reconstruct TVT
    #     from neighbour wells' formation surfaces (6 formations × 3 datum segments + dense ANCC). ---
    fcols = _formation_cols(h, ei, em, well.name, anchor, formation_ref)
    X = np.column_stack([cols[k] for k in STACK_FEATS] + [anchor, hp[em] - anchor, fusion, wt]
                        + pf_cols + a_cols + b_cols + r_cols + s_cols + fcols)
    res = dict(X=X, anchor=anchor, fusion=fusion, ff=baseline_ffill(h), em=em,
               mean_dis=float(np.mean(a_absdiff)))  # per-well HMM-PF disagreement (adaptive blend)
    if "TVT" in h.columns:
        res["truth"] = ev["TVT"].values
    return res


GBM_STEP = 5  # train on every GBM_STEP-th eval row per well (subsample-as-regularization)


def fit_gbm_stack(train_wells, level_model, step: int = GBM_STEP, seeds=(0, 1, 2)):
    """Fit the diverse-booster stack on train wells (target = TVT - level anchor).

    Phase 2 (notes/10): trains a LightGBM + CatBoost pair (two genuinely different modern
    boosters) on every `step`-th eval row per well. Both levers are lockbox-validated:
    subsampling regularizes the dense autocorrelated rows (lockbox 13.77 -> 13.51 vs full
    data), and averaging the two algorithms adds diversity (lockbox -> 13.505). HistGBM is
    dropped — it dilutes the held-out blend."""
    import lightgbm as lgb
    from catboost import CatBoostRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPRegressor
    spatial_ref = build_spatial_ref(train_wells)  # imputes each well from its neighbours (self-excluded)
    formation_ref = build_formation_ref(train_wells)  # FormationPlaneKNN + dense ANCC (notes/14)
    X, y, dis = [], [], []
    for w in train_wells:
        f = _stack_features(w, level_model, spatial_ref=spatial_ref, formation_ref=formation_ref)
        if f is None or "truth" not in f:
            continue
        X.append(f["X"][::step]); y.append((f["truth"] - f["anchor"])[::step]); dis.append(f["mean_dis"])
    X = np.vstack(X); y = np.concatenate(y)
    # Seed-ensemble (3 seeds each): removes the ~0.08 single-seed lockbox lottery + reduces variance.
    models = []
    for sd in seeds:
        models.append(lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=63,
                      min_child_samples=200, reg_lambda=1.0, subsample=0.8, subsample_freq=1,
                      colsample_bytree=0.8, random_state=sd, n_jobs=-1, verbose=-1).fit(X, y))
        models.append(CatBoostRegressor(iterations=600, learning_rate=0.03, depth=8, l2_leaf_reg=3.0,
                      rsm=0.8, random_seed=sd, thread_count=-1, verbose=0).fit(X, y))
    # Neural net (different function class) for DIVERSITY. Worse alone but blended 0.7/0.3 adds
    # -0.20 pooled CV (13.20 -> 13.00) via decorrelated errors (corr 0.64). notes/13.
    Xn = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    scaler = StandardScaler().fit(Xn)
    tm = np.arange(len(y)) if len(y) <= 200000 else np.random.default_rng(0).choice(len(y), 200000, replace=False)
    Xs = scaler.transform(Xn[tm])
    mlps = [MLPRegressor(hidden_layer_sizes=(256, 128, 64), activation="relu", alpha=1e-3,
                         batch_size=4096, learning_rate_init=1e-3, early_stopping=True,
                         n_iter_no_change=8, max_iter=120, random_state=sd).fit(Xs, y[tm])
            for sd in (0, 1)]
    return {"models": models, "mlps": mlps, "scaler": scaler, "spatial_ref": spatial_ref,
            "formation_ref": formation_ref,
            "dis_mean": float(np.mean(dis)), "dis_std": float(np.std(dis) + 1e-9)}


def _adaptive_w(mean_dis, gbm, w0=0.70, slope=0.05, lo=0.55, hi=1.0):
    """Per-well blend weight: drop the HMM/PF fusion more where HMM and PF DISAGREE (uncertain
    wells where the fusion drifts). z-scored vs train. notes/CONCLUSION: lockbox 13.547->13.512."""
    z = (mean_dis - gbm.get("dis_mean", mean_dis)) / gbm.get("dis_std", 1.0)
    return float(np.clip(w0 + slope * z, lo, hi))


def gbm_stack(well: Well, level_model, gbm, blend: float = 0.75) -> np.ndarray:
    """Blend the seed-ensembled (LightGBM+CatBoost)-corrected anchor with the variance fusion,
    using a per-well ADAPTIVE blend (less fusion where HMM/PF disagree). Our best model."""
    f = _stack_features(well, level_model, spatial_ref=gbm.get("spatial_ref"),
                        formation_ref=gbm.get("formation_ref"))
    out = baseline_ffill(well.horizontal).astype(float)
    if f is None:
        return out
    gbm_pred = np.mean([m.predict(f["X"]) for m in gbm["models"]], axis=0)
    if gbm.get("mlps"):  # 0.7 GBM + 0.3 MLP (neural diversity, -0.20 pooled; notes/13)
        Xn = np.nan_to_num(f["X"], nan=0.0, posinf=0.0, neginf=0.0)
        Xs = gbm["scaler"].transform(Xn)
        mlp_pred = np.mean([m.predict(Xs) for m in gbm["mlps"]], axis=0)
        pred = 0.7 * gbm_pred + 0.3 * mlp_pred
    else:
        pred = gbm_pred
    gbmA = f["anchor"] + pred
    out[f["em"]] = blend * gbmA + (1 - blend) * f["fusion"]  # global w=0.75 (adaptive hurt public)
    return out


# --- Particle filter ---------------------------------------------------------
# Bootstrap PF over (TVT, dip-rate): forward filtering + systematic resampling.
# Worse than the HMM alone, but a DIFFERENT algorithm on the same typewell reference,
# so it makes different errors — as a GBM-stack feature it adds ~0.24 RMSE (13.77->13.53).

def _sysresample(w: np.ndarray, rng) -> np.ndarray:
    n = len(w)
    pos = (rng.random() + np.arange(n)) / n
    return np.clip(np.searchsorted(np.cumsum(w), pos), 0, n - 1)


def pf_posterior(well: Well, n_particles: int = 600, s_target: int = 250, band_pad: float = 90.0,
                 mom: float = 0.98, sig_r: float = 0.003, sig_p: float = 0.45, lam: float = 1.0,
                 df: float = 4.0, seed: int = 0):
    """Bootstrap particle filter posterior (pred_rows, std_rows) over ALL rows."""
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

    pos = last + rng.normal(0, 3.0, n_particles)
    rate = ir + rng.normal(0, 0.01, n_particles)
    w = np.full(n_particles, 1.0 / n_particles)
    mean = np.zeros(S); var = np.zeros(S)
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
    pred = np.interp(np.arange(n), centers, mean)
    sd = np.sqrt(np.interp(np.arange(n), centers, var))
    out = ff.copy(); out[ei] = np.where(np.isfinite(pred), pred, last)
    std[ei] = np.where(np.isfinite(sd) & (sd > 0), sd, 1e6)
    return out, std
