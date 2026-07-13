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
| gbm_stack | 13.769 | 13.727 | HistGBM(HMM+level+trajectory) blend w/ fusion; notes/06 |
| gbm_stack + PF | 13.530 | 13.854 | + bootstrap particle-filter features (diverse tracker); notes/07 |
| **gbm_stack + PF + A+B** | **13.540** | — | + uncertainty (A) & GR-envelope (B) feats; kept by LOCKBOX not pooled CV; notes/08 |

- **Current best = `gbm_stack`** (`configs/gbm_stack.yaml`), now incl. PF + A+B features. **Selected by lockbox.**
- ⚠️ **A+B is a wash on pooled CV** (13.530→13.540, −0.01) but improves the held-out **lockbox**
  (154 never-tuned wells: 13.742→13.623, −0.12; lockbox-deep 16.112→15.969). Kept because the
  lockbox is the clean private-set proxy. Honest: small gain, within noise on pooled CV. Group C
  (multi-scale NCC quality) was INERT and dropped. `tools/frontier_cv.py` runs the lockbox harness;
  `_cache_base.py` + `_cache_frontier.py` rebuild the /tmp posterior caches (wiped on reboot).
- ⚠️ **CV/public diverge**: PF improves CV (13.77→13.53) but public got worse (13.73→13.85).
  Public LB = 3 leaked, non-representative wells → noise. Trust the 773-well OOF CV; it estimates
  the private (~200-well) LB that decides ranking.
- `uv run python -u tools/reproduce_cv.py` regenerates the CV table (~10 min).
- Submit (code comp = notebook only): kernel in `kernel/`, pushed as `gtwm98/rogii-hmm-fusion`;
  `kaggle competitions submit <comp> -k <kernel> -v <n> -f submission.csv -m ...`.

## ⭐ Key winning principle
**Ensemble diversity must come from a DIFFERENT ALGORITHM, not a restatement of the same one.**
The self-reference HMM (same grid-FB algo as the typewell HMM) was redundant (+0.001); the
particle filter (different algo, same reference) was complementary (+0.24). This is the one
lever that broke our plateau.

## Closed levers (don't re-investigate — see experiments/log.md + notes)
- `.png` cross-sections: TRAIN-ONLY (0 test images) → cannot be an inference feature.
- Structural geometry (datum D=TVT+Z): diverges — within-lateral Z–TVT decoupling.
- Self-correlation / NCC (self-ref HMM, self-corr emission): redundant with the typewell HMM.
- HMM resolution: finer HURTS (250-bin best). LightGBM/CatBoost: tie HistGBM (features not model).
- HMM parameter tuning: OVERFITS (holdout worse than EDA defaults).
- Tracker-diversity vein SATURATED: PF +0.24, self-PF only +0.07 (below LB noise floor, not shipped).
- Public LB is 3 leaked wells → not representative; median 9.65 is inflated by notebook forks. Trust CV.

## Where the remaining headroom is (frontier-level, large build)
Closing to the public-notebook tier (~7–9) needs their full pipeline: multi-scale NCC feeding a
LightGBM+CatBoost stack + particle-filter/HMM blend + heavy tuning. Individual levers we can bolt
onto our stack are now marginal/negative and below the public-LB noise floor — the gap is
whole-pipeline engineering, not increments.
