"""Models.

Start with a trivial, honest baseline so the whole pipeline (load → feature → predict →
submit → score) runs end to end. Beat it deliberately, one idea at a time, logging each
attempt in experiments/log.md and the vault note.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import TARGET_INPUT


def baseline_ffill(horizontal: pd.DataFrame, target_input_col: str = TARGET_INPUT) -> np.ndarray:
    """Naive baseline: carry the last known TVT_input forward into the eval zone.

    Geologically this says "the formation stays flat where we can't see it" — usually
    wrong (that's the whole point of the task), but it's a valid, submittable floor to
    measure real models against.
    """
    return horizontal[target_input_col].ffill().bfill().to_numpy()


def baseline_linear_interp(horizontal: pd.DataFrame, target_input_col: str = TARGET_INPUT) -> np.ndarray:
    """Slightly better floor: linearly interpolate TVT across the gap using MD as x."""
    s = horizontal.set_index("MD")[target_input_col].interpolate(method="index")
    return s.bfill().ffill().to_numpy()


# TODO (see vault note "Ideas backlog"):
#   - GR cross-correlation alignment of horizontal vs typewell
#   - dynamic time warping between the two GR series
#   - gradient-boosted regressor on engineered features (grouped CV)
#   - use the .png cross-section as an image channel (multimodal)
