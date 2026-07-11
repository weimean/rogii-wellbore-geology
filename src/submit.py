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
from .model import baseline_ffill, baseline_linear_interp

PREDICTORS = {
    "ffill": baseline_ffill,
    "linear_interp": baseline_linear_interp,
}


def build_submission(test_dir: Path, predictor_name: str) -> pd.DataFrame:
    """One row per EVALUATION-ZONE point only.

    Verified against sample_submission.csv: ids are ``{WELL}_{row_index}`` where
    row_index is the 0-based positional index within the horizontal_well.csv, and only
    rows with NaN ``TVT_input`` (the eval zone) are scored/submitted.
    """
    predictor = PREDICTORS[predictor_name]
    rows: list[dict] = []
    for well in load_split(test_dir):
        preds = predictor(well.horizontal)
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
    out_path = Path(cfg["output"]["submission_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sub = build_submission(test_dir, cfg["model"]["predictor"])
    sub.to_csv(out_path, index=False)
    print(f"Wrote {len(sub):,} rows -> {out_path}")


if __name__ == "__main__":
    main()
