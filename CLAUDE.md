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
- Report grouped-by-well CV RMSE. CV and the public LB agree for GR/model features (verified by a
  twin-matcher leak probe: public 13.604 == phase-2 13.601). The 3 `data/raw/test/` wells are leaked
  EXAMPLES, not the scored set. See notes/CONCLUSION.md.
- ⚠️ **EXCEPTION — spatial/offset-well features: grouped CV UNDERSTATES them massively.** Grouping by well
  strips each well's pad-neighbours into the same held-out fold, so on train only ~4.6% of wells have a
  close neighbour in the reference cloud. The scored test wells ARE offset wells to the train set, so
  spatial features fire on far more of them: the spatial-formation feature read +0.05 on grouped CV but
  **−1.11 on public** (12.948→11.839). For anything that uses neighbour wells, trust the public LB, not CV.
  notes/13.

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
| **+ neural-net (MLP) diversity** | **13.004** | **12.948** | 0.7·(LGB+Cat) + 0.3·MLP(256,128,64); different algo again; CONFIRMED on public; notes/12 |
| **+ spatial formation features (BEST, BREAKTHROUGH)** | **~12.95** | **11.839** | impute neighbour wells' structural surfaces; CV UNDERSTATED (grouped CV removes neighbours); public −1.11; kernel v14; notes/13 |
| ~~+ neighbour-TVT-direct (twin abs datum)~~ (REVERTED) | — | 11.988 | v14 already saturates the twin signal; abs-level feature added noise; kernel v15 reverted; notes/13 |
| **+ rich formation-plane features (BEST)** | — | **11.101** | top-team lever: FormationPlaneKNN (6 formations, WLS plane over k=10 neighbours) + dense ANCC + b-segments; public −0.74; kernel v16; notes/14 |
| ~~+ strengthen formation (uncertainty+cross-signals+savgol)~~ (REVERTED) | — | 11.503 | +5 feats + savgol regressed; formation-plane lever saturated; kernel v17 reverted; notes/14 |
| **+ likelihood-weighted PF ensemble (BEST)** | — | **11.042** | 8-seed PF, softmax(loglik/scale) over {3,5,8,12}; marginal −0.06; kernel v18; notes/14 |
| ~~+ fusion-weight cut (blend 0.75→0.90)~~ (REVERTED) | 10.80 | 11.072 | −0.10 grouped CV but +0.03 public (CV-overfit blend); reverted; kernel v19; notes/15 |
| **+ TabICL foundation model** | — | **11.020** | offline TabICLv2 regressor blended 0.35 (diff function class, like MLP); −0.022; kernel v20; notes/17 |
| **+ TCN sequence model (BREAKTHROUGH)** | −0.034 | **10.852** | learned HOLISTIC GR reading (dilated temporal conv over the eval sequence); blend 0.1; grouped CV UNDERSTATED it 5× (−0.034 CV → **−0.168 public**), like spatial; kernel v25; notes/27 |
| **+ TCN blend 0.1→0.2** | — | **10.760** | CV-chosen weight was too low; public −0.092; kernel v26; notes/27 |
| **+ stronger TCN (5blk/ch64/5seed) + blend 0.3 (BEST)** | −0.081 | **10.699** | stronger net doubled CV signal (−0.034→−0.081); public −0.061; gains DIMINISHING (−0.168→−0.092→−0.061); kernel v27; notes/27 |

| ~~+ transformer ensembled w/ TCN (0.65/0.35)~~ (REVERTED) | — | 10.726 | self-attention RESTATES the TCN's holistic reading (like beam-DP vs HMM); +0.027; kernel v28; notes/27 |

- **Current best = kernel v27** (`gtwm98/rogii-hmm-fusion`, ref 54832887 (BEST), stronger TCN + blend 0.3,
  public **10.699**). Sequence-model vein now SATURATED/CLOSED (−0.168→−0.092→−0.061→+0.027 transformer REJECTED).
  ffill 15.9 → 10.699 (−5.21 total). Recipe below (was v25): seed-ensembled LGB+Cat
  (3 seeds, step=5) ⊕ PF ⊕ lik-PF ⊕ A+B ⊕ **rich multi-offset GR-match features** ⊕ **MLP** ⊕ **crude spatial** ⊕
  **rich formation-PLANE features** (FormationPlaneKNN + dense ANCC) ⊕ **TabICL foundation model** (BSD-3, offline) ⊕
  **TCN sequence model** (learned holistic GR reading, torch, offline, 91 honest per-well channels),
  correction = [0.65·(0.7·GBM+0.3·MLP)+0.35·TabICL] then 0.9·(that)+0.1·TCN, final blend 0.75. **Public 10.852**
  — from ffill 15.9 (−5.06 total; −0.17 from v20). **Fully honest & rules-compliant.** Full ledger: notes/CONCLUSION.md.
- 🚀 **THE "HONEST CEILING" WAS FALSE — the TCN sequence model broke it (2026-07-19, notes/27).** After 7 levers
  closed (notes/20-26) I concluded 11.020 was the honest ceiling. WRONG. The gap was ARCHITECTURAL: every model
  predicted each row INDEPENDENTLY; a TCN reading the whole well as a SEQUENCE (the learned analogue of a
  geologist reading the log holistically) added −0.168 public. **Grouped CV understated it 5× (−0.034)** — trust
  public for the sequence model like for spatial. LIKELY MORE HEADROOM: the blend weight (0.1, set by the
  understating CV) is probably too low; a stronger TCN/transformer may add more. Tune the blend on PUBLIC.
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
