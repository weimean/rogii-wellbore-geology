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
