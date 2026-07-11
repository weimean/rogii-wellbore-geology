"""Load and join per-well competition files.

Each well = one horizontal_well.csv (the sequence we predict along) plus one
typewell.csv (the vertical reference log we correlate against). Files are keyed by an
8-char hash, e.g. ``015fe0d2``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# Column groups from the competition data description.
HORIZONTAL_TRAJECTORY = ["MD", "X", "Y", "Z"]
FORMATION_DEPTHS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]  # train only
TARGET = "TVT"          # train only — the label
TARGET_INPUT = "TVT_input"  # feature copy of TVT, NaN inside the evaluation zone
LOG = "GR"


@dataclass
class Well:
    """One well's paired data."""

    name: str
    horizontal: pd.DataFrame
    typewell: pd.DataFrame

    @property
    def eval_mask(self) -> pd.Series:
        """Rows in the evaluation zone = where TVT_input is NaN (what we must predict)."""
        return self.horizontal[TARGET_INPUT].isna()


def well_names(split_dir: Path) -> list[str]:
    """Discover well hashes in a train/ or test/ directory."""
    return sorted(
        p.name.split("__")[0]
        for p in split_dir.glob("*__horizontal_well.csv")
    )


def load_well(split_dir: Path, name: str) -> Well:
    horizontal = pd.read_csv(split_dir / f"{name}__horizontal_well.csv")
    typewell = pd.read_csv(split_dir / f"{name}__typewell.csv")
    horizontal["WELLNAME"] = name
    typewell["WELLNAME"] = name
    return Well(name=name, horizontal=horizontal, typewell=typewell)


def load_split(split_dir: Path, limit: int | None = None) -> list[Well]:
    """Load every well in a split. ``limit`` is handy for quick iteration."""
    names = well_names(split_dir)
    if limit is not None:
        names = names[:limit]
    return [load_well(split_dir, n) for n in names]


def submission_id(well: str, row_index: int) -> str:
    """id format required by sample_submission.csv: ``{WELLNAME}_{row_index}``."""
    return f"{well}_{row_index}"
