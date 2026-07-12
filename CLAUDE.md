# ROGII Wellbore Geology Prediction

Kaggle Featured Code Competition — predict **TVT** (True Vertical Thickness, ft) for the
post-Prediction-Start evaluation zone of horizontal wells. Metric: **RMSE**.
Final deadline: **Aug 5, 2026**.

## Hard constraints
- Notebook-only submission, ≤9h CPU/GPU, internet disabled.
- Output must be `submission.csv` with columns `id, tvt` (eval-zone rows only).
- External data / pre-trained models allowed if uploaded as a dataset.

## Repo layout
```
src/           data.py  features.py  validation.py  model.py  submit.py
configs/       one YAML per experiment
notes/         working notes (EDA, per-approach analysis via the rubric)
experiments/   log.md — terse per-run ledger (CV RMSE, keep/drop)
data/raw/      train/ (773 wells × 3 files each) · test/ (3 example wells)
```

## Key domain facts
- Eval zone = contiguous NaN tail of TVT_input (~74% of well on median).
- Core signal: GR cross-correlation — horizontal GR is a shifted copy of the
  vertical typewell GR. Self-correlation (own pre-PS GR) > typewell > offset wells.
- Trajectory columns: MD, X, Y, Z. Formation tops (ANCC…BUDA) are train-only.
- Typewell has columns: TVT, GR, Geology.

## Validation discipline
- **Always split by well** (`GroupKFold` on `WELLNAME`). Never leak rows across folds.
- Report grouped-by-well CV RMSE. Trust CV over the stub public LB.

## Workflow
- One config YAML per experiment; log every run in `experiments/log.md`.
- Per-approach narrative in `notes/approaches/NN-name.md` (idea → validation → conclusion).
- Manage deps with `uv sync`; run with `uv run python -m src.submit --config <cfg>`.

## Model progression (773-well GroupKFold CV; ffill=15.910)
| Predictor (config) | CV RMSE | Public LB | Note |
|----|----|----|----|
| ffill | 15.910 | — | flat last-known TVT |
| level_regression | 15.325 | — | Ridge on per-well offset (pre-PS velocity + GR); notes/03 |
| hmm_fusion | 14.239 | 15.116 | 2nd-order (TVT,dip-rate) HMM ⊕ level by posterior var; notes/05 |
| **gbm_stack (BEST)** | **13.769** | **13.727** | HistGBM(HMM+level+trajectory) 0.5-blend w/ fusion; notes/06 |

- **Current best = `gbm_stack`** (`configs/gbm_stack.yaml`). Public rank ~3353/4750.
- `uv run python tools/reproduce_cv.py` regenerates the whole CV table (~8 min).
- Submit (code comp = notebook only): kernel in `kernel/`, pushed as `gtwm98/rogii-hmm-fusion`;
  `kaggle competitions submit <comp> -k <kernel> -v <n> -f submission.csv -m ...`.

## Closed levers (don't re-investigate — see experiments/log.md + notes)
- `.png` cross-sections: TRAIN-ONLY (0 test images) → cannot be an inference feature.
- Structural geometry (datum D=TVT+Z): diverges — within-lateral Z–TVT decoupling.
- Self-correlation / NCC (self-ref HMM, self-corr emission): redundant with the typewell HMM.
- HMM resolution: finer HURTS (250-bin best). LightGBM/CatBoost: tie HistGBM (features are the bottleneck).
- Public LB is 3 leaked wells → not representative; median 9.65 is inflated by notebook forks. Trust CV.

## Where the remaining headroom is (frontier-level, large build)
Closing to the public-notebook tier (~7–9) needs their full pipeline: multi-scale NCC feeding a
LightGBM+CatBoost stack + particle-filter/HMM blend + heavy tuning. Individual levers we can bolt
onto our HMM-centric stack are marginal/negative — the gap is whole-pipeline engineering.
