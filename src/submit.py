"""Build a submission.csv in the required format.

    id  = {WELLNAME}_{row_index}
    tvt = predicted True Vertical Thickness

Only rows in the evaluation zone (TVT_input is NaN) need predicting, but we emit a value
for every row the sample_submission asks for. Run:

    uv run python -m src.submit --config configs/baseline.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from .data import TARGET_INPUT, load_split, submission_id
from .model import (
    baseline_ffill,
    baseline_linear_interp,
    kalman_gr,
    gr_offset_search,
    ensemble_kalman_offset,
    fit_level_model,
    level_regression,
    hmm_fusion,
    fit_gbm_stack,
    gbm_stack,
)

SIMPLE_PREDICTORS = {
    "ffill": baseline_ffill,
    "linear_interp": baseline_linear_interp,
}

WELL_PREDICTORS = {
    "kalman_gr": kalman_gr,
    "gr_offset": gr_offset_search,
    "ensemble": ensemble_kalman_offset,
}

# Predictors that must be FIT on the training split before predicting on test.
TRAINED_PREDICTORS = {"level_regression", "hmm_fusion", "gbm_stack"}


def build_submission(test_dir: Path, predictor_name: str, train_dir: Path | None = None) -> pd.DataFrame:
    """One row per EVALUATION-ZONE point only."""
    pred_fn = None
    if predictor_name in TRAINED_PREDICTORS:
        if train_dir is None:
            raise ValueError(f"{predictor_name} needs a train_dir to fit on")
        train_wells = load_split(train_dir)
        model = fit_level_model(train_wells)
        if predictor_name == "hmm_fusion":
            pred_fn = lambda well: hmm_fusion(well, model)  # noqa: E731
        elif predictor_name == "gbm_stack":
            gbm = fit_gbm_stack(train_wells, model)
            pred_fn = lambda well: gbm_stack(well, model, gbm)  # noqa: E731
        else:
            pred_fn = lambda well: level_regression(well, model)  # noqa: E731
    elif predictor_name in WELL_PREDICTORS:
        pred_fn = WELL_PREDICTORS[predictor_name]

    rows: list[dict] = []
    for well in load_split(test_dir):
        if pred_fn is not None:
            preds = pred_fn(well)
        else:
            preds = SIMPLE_PREDICTORS[predictor_name](well.horizontal)
        eval_mask = well.eval_mask.to_numpy()
        for row_index, (pred, is_eval) in enumerate(zip(preds, eval_mask)):
            if is_eval:
                rows.append({"id": submission_id(well.name, row_index), "tvt": pred})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    test_dir = Path(cfg["data"]["test_dir"])
    train_dir = Path(cfg["data"].get("train_dir", "data/raw/train"))
    out_path = Path(cfg["output"]["submission_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sub = build_submission(test_dir, cfg["model"]["predictor"], train_dir=train_dir)
    sub.to_csv(out_path, index=False)
    print(f"Wrote {len(sub):,} rows -> {out_path}")


if __name__ == "__main__":
    main()
