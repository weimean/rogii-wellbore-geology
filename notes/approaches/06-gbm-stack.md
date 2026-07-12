# Approach 06 — GBM stack over HMM + level + trajectory

**Status:** concluded-kept (current best, 13.4% over ffill)
**Date:** 2026-07-12
**Config / commit:** `configs/gbm_stack.yaml` (predictor `gbm_stack`)

## 1. Idea & motivation
First LB submission (hmm_fusion, public 15.116, rank ~3626/4734, median 9.65) showed we're
behind the field — the wiggle is very recoverable and our single HMM+Ridge is under-built
vs the frontier's GBM stacks. Push toward the field's method: stack the HMM posterior with
the level anchor and trajectory features in a gradient-boosted tree that learns a residual
correction, then blend with the hand-tuned variance fusion.

## 2. Method
- **Features per eval row** (`_stack_features`): HMM posterior mean + std; level anchor;
  hmm−anchor; variance fusion + its weight; z_from_ps, md_from_ps, horiz_from_ps, dz,
  tortuosity (rolling std of dZ), gr50 (smoothed GR), gr_dev; signed-azimuth sin/cos;
  eval_frac; v50, v200, gr_off, known_range, eval_len.
- **Model** (`fit_gbm_stack`): HistGradientBoostingRegressor (600 iters, lr 0.03, 63 leaves,
  min_leaf 200, L2 1.0) predicting **TVT − level_anchor** (a residual correction).
- **Prediction** (`gbm_stack`): final = 0.5·(anchor + GBM_correction) + 0.5·variance_fusion.
  The two are complementary — the blend beats either alone.

## 3. Validation — full 773-well GroupKFold (from cached HMM+features)
| Model | RMSE | Δ vs ffill |
|----|----|----|
| ffill | 15.910 | — |
| level_regression | 15.325 | −3.7% |
| hmm_fusion | 14.216 | −10.7% |
| GBM (correct anchor) alone | 14.162 | −11.0% |
| **0.5·GBM + 0.5·fusion** | **13.769** | **−13.4%** |
Held-out end-to-end sanity (fit 140 / predict 30 disjoint): gbm_stack 9.84 vs hmm_fusion 10.99.

## 4. What worked / what didn't
- **Worked:** GBM correcting the *anchor* (not the fusion) + 0.5 blend with the fusion.
- **Didn't:** GBM correcting the fusion directly (13.87 > 13.77); self-correlation emission
  in the HMM (regression to 15.37 — GR calibration distorts the emission scale; reverted).
- **Lesson:** A/B only against the true production baseline on a representative sample —
  the self-correlation A/B on 200 wells was misleading because both arms had calibration.

## 5. Uncertainty
- HistGBM used locally (libomp missing for LightGBM); kernel uses the same HistGBM.
- Blend weight 0.5 and GBM hyperparams mildly tuned on CV; landscape is flat.
- Still far from the field median (9.65). The untapped big lever is **multi-scale NCC
  self-correlation features** (romantamrazov) — not yet built (needs a cache rebuild).

## 6. Conclusion
Stacking the HMM with a GBM and blending with the variance fusion moved CV 14.24 → 13.77
(ffill→−13.4%). Efficient iteration enabled by caching HMM posteriors + features once
(`/tmp/rogii_feats.npz`, 3.78M rows). Next: multi-scale NCC self-correlation features,
then LightGBM+CatBoost in the kernel where they're available.
