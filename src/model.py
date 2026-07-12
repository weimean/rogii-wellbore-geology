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
