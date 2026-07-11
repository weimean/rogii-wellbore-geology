"""Feature engineering.

The core signal is *log correlation*: the horizontal GR curve is a shifted/stretched
view of the vertical typewell GR signature. Features here are deliberately simple
starting points — the real modelling work (dynamic time warping, cross-correlation
alignment, using the .png cross-section) goes on the Ideas backlog in the vault note.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import LOG, TARGET_INPUT


def add_trajectory_features(horizontal: pd.DataFrame) -> pd.DataFrame:
    """Local geometry of the wellbore path: dip and step size along MD."""
    df = horizontal.copy()
    df["dZ"] = df["Z"].diff()
    df["dMD"] = df["MD"].diff()
    # inclination proxy: vertical change per unit measured depth
    df["dip"] = (df["dZ"] / df["dMD"]).replace([np.inf, -np.inf], np.nan)
    return df


def add_gr_features(horizontal: pd.DataFrame, windows: tuple[int, ...] = (5, 15, 31)) -> pd.DataFrame:
    """Rolling statistics of the Gamma-Ray log — smooth out noise, expose shape."""
    df = horizontal.copy()
    for w in windows:
        df[f"GR_mean_{w}"] = df[LOG].rolling(w, center=True, min_periods=1).mean()
        df[f"GR_std_{w}"] = df[LOG].rolling(w, center=True, min_periods=1).std()
        df[f"GR_grad_{w}"] = df[f"GR_mean_{w}"].diff()
    return df


def known_tvt_anchors(horizontal: pd.DataFrame) -> pd.DataFrame:
    """Rows where TVT is known (outside the eval zone) — anchors for interpolation
    or for aligning against the typewell."""
    return horizontal[horizontal[TARGET_INPUT].notna()]


def build_features(horizontal: pd.DataFrame) -> pd.DataFrame:
    """Compose the baseline feature set for one well."""
    df = add_trajectory_features(horizontal)
    df = add_gr_features(df)
    return df
