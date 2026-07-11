# ROGII — Wellbore Geology Prediction

Kaggle Featured **Code Competition** · predict **TVT (True Vertical Thickness, ft)**
for the evaluation zone of each horizontal well. Metric: **Mean Squared Error (MSE)**.

- Competition: <https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction>
- 📓 **Knowledge / decisions / experiment log:** Obsidian vault note
  `04 Projects/Kaggle — ROGII Wellbore Geology Prediction.md`
  (this repo holds code; the vault note is the source of truth for *why* decisions were made)

## The problem in one paragraph

Each horizontal well has a Gamma-Ray (`GR`) log along measured depth (`MD`) plus a
trajectory (`X`, `Y`, `Z`). A separate **typewell** gives the vertical `GR` signature
tagged with geological formations. The target `TVT` (the well's geological position) is
known everywhere *except* an **evaluation zone** where it is `NaN`. The task is
essentially **log correlation / sequence alignment**: infer `TVT` in the gap by matching
the horizontal `GR` against the vertical typewell `GR`. Each well also ships a `.png`
cross-section (the "multimodal" channel — optional to use).

## Quickstart

```bash
uv sync                              # create env from pyproject.toml + uv.lock
# download data (needs your Kaggle API token) — see data/README.md
kaggle competitions download -c rogii-wellbore-geology-prediction -p data/raw
unzip data/raw/*.zip -d data/raw
```

Then open `notebooks/01_eda.ipynb`, or build a baseline submission:

```bash
uv run python -m src.submit --config configs/baseline.yaml
```

## Layout

```
data/        raw + processed competition data          (gitignored)
notebooks/   01_eda.ipynb (explore) · 99_submission.ipynb (submit)
src/         data.py features.py validation.py model.py submit.py
configs/     experiment configs (yaml)
models/      trained checkpoints                        (gitignored)
outputs/     submissions, logs                          (gitignored)
experiments/ log.md — terse per-run log (mirrors the vault note's table)
```

## Validation discipline ⚠️

Split **by well** (`GroupKFold` on `WELLNAME`) — never let rows from the same well fall
into both train and validation, or CV MSE will be optimistically biased. The public test
folder here is only a few *leaked* training wells; the real hidden test (~200 wells) is
swapped in at rerun. Trust grouped CV over this stub leaderboard.
