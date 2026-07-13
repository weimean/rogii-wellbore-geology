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
| gbm_stack + PF + A+B | 13.540 | — | + uncertainty (A) & GR-envelope (B) feats; kept by lockbox; notes/08 |
| **+ LGB/Cat @ step=5 (BEST)** | **13.520** | — | LightGBM+CatBoost avg on 1/5-subsampled rows (drop HistGBM); notes/10 |

- **Current best = `gbm_stack`** (`configs/gbm_stack.yaml`): LightGBM+CatBoost avg (step=5 rows) ⊕ PF ⊕ A+B.
- **Phase 2 (notes/10):** two levers — subsample-as-regularization (train every 5th eval row) + model
  diversity (LGB+Cat, HistGBM dropped) — give canonical CV 13.540→**13.520** (−0.02, honest/unbiased).
  Lockbox showed −0.265 but that's SELECTION-BIASED (best-of-7 blends picked on it); trust the −0.02.
  Modest, directionally positive, our best config. Needs `lightgbm`+`catboost` (both on Kaggle image).
- ⚠️ **~9 is NOT reachable** (notes/09): it's a per-well LEVEL problem needing offset R²≈0.95; the full
  stack recovers 0.337 and 4 independent level levers die at ~0.10 (missing signal is geology absent
  from X/Y/Z/GR/typewell). Phase-2 gains are wiggle-side (the 32% of budget that isn't level).
- ⚠️ **A+B is a wash on pooled CV** (−0.01) but improves the lockbox; Group C (NCC quality) was INERT.
  `tools/frontier_cv.py` = lockbox harness; `tools/frontier_fast.py` = fast cached harness (step subsample);
  `_cache_base.py`+`_cache_frontier.py` rebuild the /tmp caches (wiped on reboot).
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
