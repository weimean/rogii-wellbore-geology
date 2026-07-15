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
- Report grouped-by-well CV RMSE. **CV and the public LB agree and are both trustworthy** — verified
  by submitting a twin-matcher leak probe (public 13.604 == phase-2 13.601 → no exploitable leak in the
  scored test). The 3 `data/raw/test/` wells are leaked EXAMPLES, not the scored set. See notes/CONCLUSION.md.

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
| ~~+ typewell Geology~~ (REVERTED) | ~~12.985~~ | 13.927 | LEAKAGE — Geology is TRAIN-ONLY; feature absent in test → public got worse; notes/11 |
| + seed-ensemble | 13.499 | 13.549 | LGB+Cat × 3 seeds each averaged; removes seed lottery; notes/CONCLUSION |
| + blend 0.70 | 13.472 | 13.368 | re-tuned final blend; notes/CONCLUSION |
| + blend 0.75 | 13.468 | 13.288 | best public before the ceiling break |
| ~~+ adaptive blend~~ (REVERTED) | ~~13.453~~ | 13.407 | train CV/lockbox liked it, public got WORSE; reverted to global w=0.75 |
| **+ rich features (BREAKTHROUGH)** | **13.201** | — | multi-offset GR-match grids (top-team feature family); refuted the "ceiling"; notes/12 |
| **+ neural-net (MLP) diversity (BEST)** | **13.004** | **12.948** | 0.7·(LGB+Cat) + 0.3·MLP(256,128,64); different algo again; CONFIRMED on public; notes/12 |

- **Current best = `gbm_stack`** (`configs/gbm_stack.yaml`): seed-ensembled LGB+Cat (3 seeds, step=5) ⊕ PF ⊕ A+B ⊕ **rich multi-offset GR-match features** ⊕ **MLP**, final blend 0.75. **Pooled CV 13.004 / public 12.948** (kernel v13, ref 54704906). −0.34 public vs the prior 13.288 — the ceiling break holds on the hidden test. See notes/approaches/12.
- ⚠️ **The "13.52 ceiling" was FEATURE-POVERTY, not input-limited.** I declared an input-limited ceiling
  multiple times based on testing features IN ISOLATION (each new feature ≈ +0.01). WRONG. Reproducing the
  top teams' rich feature *family* (multi-offset GR-match grids `tda/tdpf/tdhmm{offset}` = horizontal GR
  minus typewell-GR sampled at last_TVT+offset, across 11 offsets) in AGGREGATE + a strong model cleared
  −0.27 (13.47→13.20). Then a neural net (different algo) cleared another −0.20 (13.20→13.00). **Lesson:
  never conclude a ceiling from per-feature probes — test the full rich set with a strong model.**
- 🚨 **Geology was LEAKAGE, REVERTED (notes/11):** the typewell `Geology` column is **TRAIN-ONLY** —
  test typewells have only `TVT, GR` (verified: all 3 example test typewells lack the column). The
  −0.53 CV/lockbox gain was real *on train* but the feature is **absent at inference** (`_geology_features`
  falls back to zeros on test) → public got WORSE (13.601→13.927). CV+lockbox both used train wells so
  neither could catch it; only the real-test public LB did. Classic train-only-feature leak. Diagnostics:
  `tools/critique_diag.py` (bootstrap CI, ablation, test-well representativeness caught the fallback).
  **Lesson: verify a feature EXISTS at inference before building on it** — check `data/raw/test` columns.
- **Phase 2 (notes/10):** subsample-as-regularization (train every 5th eval row) + LGB/Cat diversity
  (drop HistGBM): canonical 13.540→13.520, public 13.601 (best). Needs `lightgbm`+`catboost` (Kaggle image).
- `uv run python -u tools/reproduce_cv.py` regenerates the CV table (~15 min). `tools/frontier_fast.py`
  = fast cached harness; `tools/frontier_cv.py` = lockbox; `_cache_base.py` rebuilds /tmp caches (wiped on reboot).
- Submit: kernel `kernel/` → `gtwm98/rogii-hmm-fusion` (v7 = phase-2). `kaggle competitions submit <comp>
  -k <kernel> -v <n> -f submission.csv -m ...`.

## ⚠️ THE "CEILING" WAS FALSE — it was feature-poverty (see notes/approaches/12)
**There is no 13.52 ceiling.** I repeatedly declared "13.52 is the input-limited ceiling" from
per-feature probes (each new feature ≈ +0.01, error budget 68% LEVEL / 15% trend / 18% wiggle). That
reasoning was WRONG: individual features are weak, but the top teams' rich feature **family** in
aggregate + a strong model is not. Reproducing multi-offset GR-match grids cleared −0.27 (13.47→13.20),
then a neural net another −0.20 (13.20→13.00). Real headroom remains toward the top teams' honest OOF
~11.1 (their public ~7 has an extra ~4-pt leak-flavored boost we do NOT chase). **Method that beat the
"ceiling": stop probing features one at a time — build the whole rich family and let the model use it.**
- Still true: level/trend ARE mostly geology-locked and Geology/tops are TRAIN-ONLY (don't build on them
  as labels). But the GR *signal itself* (typewell GR vs horizontal GR, sampled across many TVT offsets)
  is available at inference and carries far more than the isolated probes suggested.

## Key winning principle
Ensemble diversity must come from a **DIFFERENT ALGORITHM**, not a restatement. Self-ref HMM (same algo)
= redundant (+0.001); the particle filter (different algo) = complementary (+0.24). LGB+Cat over HistGBM
same idea. The **MLP** (neural net, different function class) added −0.20 pooled on top of LGB+Cat even
though it's worse alone (corr 0.64). Subsampling eval rows regularizes (the dense rows overfit).
**AND: build feature FAMILIES, not one-off features** — a family of 40+ multi-offset GR-match columns
cleared −0.27 where each isolated probe read ≈ +0.01. The ceiling I kept declaring was feature-poverty.

## Closed levers (don't re-investigate — see notes/CONCLUSION.md)
- Level: velocity/spatial-KNN/GR-alignment/global-datum all die at offset R²≈0.10 (`level_probe`, `level_align`).
- Trend/slope: R²≈0 (`slope_probe`). Geology labels: strong but TRAIN-ONLY leak, reverted (notes/11);
  inferring from GR is marginal (0.113). Formation tops: TRAIN-ONLY; oracle +0.014 (`tops_probe`).
- Geosteering trajectory×GR, uncertainty shrinkage: residual-R²≈0, no gain (`residual_diag`, `shrink_test`).
- `.png`: train-only. Self-correlation, HMM resolution/tuning: redundant/overfit. Datum U=T+Z: catastrophic (`datum_test`).
- **Leak: NONE exploitable in the scored test** (twin-probe = phase-2; `kernel_leakprobe/`). Public LB is trustworthy.
