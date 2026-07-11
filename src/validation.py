"""Cross-validation and scoring.

⚠️ Split BY WELL. Rows within a well are highly autocorrelated; if the same well appears
in both train and validation folds, CV RMSE is optimistically biased and will not track
the hidden leaderboard.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold


def group_folds(df: pd.DataFrame, n_splits: int = 5, group_col: str = "WELLNAME"):
    """Yield (train_idx, valid_idx) grouped by well. df must carry ``group_col``."""
    gkf = GroupKFold(n_splits=n_splits)
    groups = df[group_col].values
    yield from gkf.split(df, groups=groups)


def score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Competition metric: Root Mean Squared Error (matches the Kaggle leaderboard).

    The 'Mean Squared Error' Evaluation *tag* is imprecise; the Evaluation section gives
    the RMSE formula explicitly. Ranking is identical either way.
    """
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))
