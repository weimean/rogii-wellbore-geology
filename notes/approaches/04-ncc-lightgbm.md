# Approach 04 — NCC alignment + GBM (leaderboard-informed)

**Status:** concluded-negative (ties level_regression on honest CV; kept as feature foundation)
**Date:** 2026-07-12

## 1. Idea & motivation
Web research (see memory `leaderboard-intel`) of the public LB + mycarta/rogii-geosteering-
toolkit suggested the within-well TVT curve (the "wiggle") is recoverable via multi-scale
normalized cross-correlation (NCC) of horizontal GR vs typewell, fed into LightGBM with
domain features (Q-3D tortuosity, signed azimuth). Top public LB = 4.86 vs our CV 15.27.

## 2. Method
- **NCC aligner** (prototype): per eval window, search (TVT offset, slope) maximizing
  normalized cross-correlation of smoothed horizontal GR vs typewell GR at candidate TVT.
- **GBM** (HistGradientBoostingRegressor; LightGBM unavailable locally — needs libomp, but
  is available in the Kaggle kernel): per-eval-row, target = TVT − last_known. Features:
  z_from_ps, md_from_ps, horiz_from_ps, dz, tortuosity (rolling std of dZ), multi-scale
  smoothed GR (σ=15/50/150), gr_dev, gr_vs_typewell@ffill, gr_off (calibrated offset),
  v50/v200 velocity, signed-azimuth sin/cos, eval_frac, last_tvt, known_range, eval_len.
- CV: GroupKFold(5) by WELLNAME (leak-free, comparable to prior approaches).

## 3. Validation (global RMSE, GroupKFold)
| Model | RMSE |
|----|----|
| ffill | 15.910 |
| level_regression (Approach 03) | 15.266 |
| **HistGBM (this approach)** | **15.247** |
| NCC alignment standalone | 21.691 (worse than ffill — see §5) |

Feature importance (permutation): **z_from_ps 0.39** (dominant), v200 0.05, gr_off 0.045,
horiz_from_ps 0.035, last_tvt 0.034, v50 0.025, az_cos 0.020, gr150 0.007, gr15 0.001.

## 4. Key conclusion — the public LB is a leaked-well artifact
The 3 public test wells are **byte-identical to their train twins** (GR identical, same
eval mask; train has true TVT) → the public LB (top 4.86) is lookup-gameable and only 3
non-representative wells. On our honest 773-well GroupKFold CV, the full NCC+GBM+domain
approach (**15.247**) merely **ties** the simple level_regression (15.266). Every method we
have tried clusters at CV ≈ 15.2–15.9. mycarta's reported feature deltas (−0.107 tortuosity,
+0.476 AEON) are consistent with this same ~15 scale — small gains on a genuinely hard
problem. **The wiggle is not recoverable to 5–6 on the real distribution; the LB misled us.**

## 5. Why NCC alignment fails standalone
The horizontal/typewell GR signature is quasi-periodic, so NCC has multiple near-equal peaks
(mean best-window correlation 0.85 but at the *wrong* absolute TVT). Independent per-window
search loses continuity and locks onto aliased offsets → 21.7 RMSE. As a GBM *feature* it
adds almost nothing beyond gr_off (the GR channel is simply weak here: SNR≈0.02 per step).

## 6. What the GBM did find
The strongest per-row signal is **z_from_ps** (trajectory), not GR — the GBM extracts a
trajectory-based level/trend. Signed azimuth (az_cos) contributes, matching mycarta's
"updip vs downdip" insight. But the net gain over the linear level model is within noise.

## 7. Verdict & next steps
- **Keep level_regression as the production best** (15.266; far simpler, ties the GBM).
- The GBM feature scaffold is a foundation if we later get a *real* signal (e.g. bed dip
  from the .png cross-sections, or a proper Q-3D tortuosity).
- Do NOT chase the public LB — trust the 773-well GroupKFold CV.
- Highest-leverage remaining idea unchanged: extract structural dip from the cross-section
  images to sharpen the LEVEL (the 68% component), which no tabular feature predicts well.
