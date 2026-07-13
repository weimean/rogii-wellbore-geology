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
| + LGB/Cat @ step=5 | 13.520 | 13.601 | LightGBM+CatBoost avg on 1/5-subsampled rows (drop HistGBM); notes/10 |
| **+ typewell Geology (BEST)** | **12.985** | 13.927 | formation-label LEVEL feats; −0.53 CV but public DIVERGES (+0.33); notes/11 |

- **Current best = `gbm_stack`** (`configs/gbm_stack.yaml`): LightGBM+CatBoost avg (step=5) ⊕ PF ⊕ A+B ⊕ Geology.
- **⭐ Geology breakthrough (notes/11):** the typewell `Geology` column (formation labels vs TVT) cracks the
  LEVEL component. A lateral holds in a target formation whose absolute TVT the typewell gives; formations
  are ordered in TVT so they DON'T alias like GR. 3 feats (geo_last_off/geo_hmm_off/geo_last_med, in
  `_geology_features`) → canonical CV 13.520→**12.985 (−0.53)**, pooled −0.49, lockbox −0.62 — all agree,
  no selection bias. Orthogonal to the HMM (GR alignment ≠ formation labels). **Geology is the one
  geological signal available at test time** (formation tops & .png are train-only).
  ⚠️ **CV/public DIVERGE hard**: CV −0.53 but public 13.601→13.927 (+0.33). BUT the lockbox
  (154 never-tuned wells) also gave −0.62, so it's not CV-fold overfitting — two independent
  held-out estimates agree. Public = 3 leaked wells → trust the CV. **Hedge: select BOTH v6
  (geology, CV-best) and v5 (phase-2, public-best) as the 2 final submissions.**
- **Phase 2 (notes/10):** subsample-as-regularization (train every 5th eval row) + LGB/Cat diversity
  (drop HistGBM): canonical 13.540→13.520. Needs `lightgbm`+`catboost` (both on Kaggle image).
- ⚠️ **~9 not yet reached but level is PARTLY recoverable now** (notes/09 amended by notes/11): the 4 dead
  level levers were all GR cross-correlation (alias); the Geology *label* channel was untested and works.
  Remaining level room: a GR→formation classifier for the eval zone (test-legal 0.187 → oracle 0.245).
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
