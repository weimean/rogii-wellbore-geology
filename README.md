# ROGII — Wellbore Geology Prediction

Kaggle Featured **Code Competition** · predict **TVT (True Vertical Thickness, ft)**
for the evaluation zone of each horizontal well. Metric: **RMSE**.
Constraints: notebook-only, **≤ 9 h** CPU/GPU, **internet disabled**, output `submission.csv`.
Final deadline **Aug 5, 2026**. External data / pre-trained models allowed if uploaded as a dataset.

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

Then build the best submission (fits on train, predicts test):

```bash
uv run python -m src.submit --config configs/gbm_stack.yaml   # current best, CV 13.77
uv run python tools/reproduce_cv.py                           # regenerate the CV table (~8 min)
```

**Current best: `gbm_stack`** — a HistGBM over the 2nd-order HMM posterior + level anchor +
trajectory, blended 0.5/0.5 with the variance fusion. CV 13.769, public LB 13.727.
Progression: ffill 15.91 → level_regression 15.33 → hmm_fusion 14.24 → **gbm_stack 13.77**.
See `notes/approaches/` (00–06) and `experiments/log.md`.

Submitting (this is a **notebook-only** code competition — direct CSV upload is rejected):
the self-contained kernel lives in `kernel/`; push with `kaggle kernels push -p kernel`, then
`kaggle competitions submit rogii-wellbore-geology-prediction -k <user>/<kernel> -v <n> -f submission.csv -m "..."`.

## Layout

```
data/        raw + processed competition data          (gitignored)
notebooks/   01_eda.ipynb (explore) · 99_submission.ipynb (submit)
src/         data.py features.py validation.py model.py submit.py
configs/     experiment configs (yaml)
kernel/      self-contained Kaggle submission script + metadata
tools/       reproduce_cv.py — regenerate the CV table
models/      trained checkpoints                        (gitignored)
outputs/     submissions, logs                          (gitignored)
experiments/ log.md — terse per-run log (mirrors the vault note's table)
```

## Validation discipline ⚠️

Split **by well** (`GroupKFold` on `WELLNAME`) — never let rows from the same well fall
into both train and validation, or CV MSE will be optimistically biased. The public test
folder here is only a few *leaked* training wells; the real hidden test (~200 wells) is
swapped in at rerun. Trust grouped CV over this stub leaderboard.
