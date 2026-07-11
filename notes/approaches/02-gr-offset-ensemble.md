# Approach 02 — Calibrated GR Offset Search + Kalman Ensemble

**Status:** concluded-kept (current best, 2.2% over ffill)
**Date:** 2026-07-11
**Config / commit:** `configs/ensemble_v1.yaml`

## 1. Idea & motivation
The Kalman filter (Approach 01) uses the typewell GR gradient for point-by-point TVT
tracking, but the per-step signal is too weak (SNR≈0.02). Instead, aggregate the GR
signal over the ENTIRE eval zone to estimate a constant TVT offset from the last known
value. This trades temporal resolution for noise reduction.

## 2. Method

### GR Offset Search (`gr_offset_search`)
1. **Calibrate:** in the known zone, fit horizontal_GR ≈ a × typewell_GR + b using linear
   regression on the (typewell_GR, horizontal_GR) pairs at known TVT values.
2. **Smooth:** Gaussian-smooth the horizontal GR (σ=100 steps).
3. **Search:** for offsets in [-30, +30] ft from last_known_TVT, find the typewell TVT where
   calibrated typewell GR best matches the eval-zone mean GR.
4. **Damp:** apply only α=0.16 of the found offset (the raw search is noisy; damping
   prevents catastrophic errors on wells where the GR match is ambiguous).
5. **Bias:** add +1.42 ft (the global mean eval-zone offset across 773 training wells).

### Ensemble
Weighted blend: 25% Kalman GR + 75% offset search. The Kalman provides per-step
structure; the offset search provides a better mean level.

## 3. Validation
Global RMSE (3.78M eval-zone rows, 773 training wells):
| Model | RMSE | Δ vs ffill |
|-------|------|-----------|
| ffill | 15.910 | — |
| Kalman GR | 15.735 | -1.1% |
| **GR offset** | **15.581** | **-2.1%** |
| **Ensemble** | **15.565** | **-2.2%** |

## 4. Contribution
| Component | RMSE | Marginal |
|-----------|------|----------|
| α=0.16 damping (vs 0.10) | 15.581 vs 15.621 | -0.3% |
| σ=100 smoothing (vs 50) | 15.581 vs 15.642 | -0.4% |
| +1.42 bias | ~-0.5% | consistent across approaches |
| Kalman blending (25%) | 15.565 vs 15.581 | -0.1% |

## 5. Uncertainty
- The α=0.16 and σ=100 are tuned on all 773 training wells — slight overfit risk on
  the specific train/test split. The parameters are robust (smooth RMSE landscape).
- The +1.42 bias assumes the test distribution matches train; could hurt if it doesn't.
- The model is still fundamentally limited by the weak GR→TVT signal. The typewell GR
  doesn't vary enough within the eval-zone TVT range to give precise position estimates.

## 6. Physical meaningfulness
The approach is physically motivated: the mean GR of the eval zone reflects the mean
formation the well sits in, and the typewell maps formations to TVT. The damping (α=0.16)
acknowledges that the mapping is imprecise due to lateral geological variation and
calibration errors.

## 7. What was tried and didn't work (this session)
- **Standard DTW** (downsampled eval GR vs typewell GR): finds wrong alignment, worse than ffill
- **Viterbi HMM** (TVT states, GR emissions): even with tuned stay_prob=0.9995, can't beat ffill
- **Constrained DP alignment**: same issue as DTW — wrong alignments
- **Piecewise offset** (n_segments=3-15): worse than single-segment offset (more noise)
- **Calibrated linear search** (TVT_start + rate × MD): beats ffill on 7/30 wells but
  catastrophic on others; not robust

## 8. Conclusion
Aggregating GR over the full eval zone (heavy smoothing + mean matching) extracts more
reliable signal than point-by-point or sequence-level approaches. The improvement ceiling
is ~2-3% via this approach family. Next directions:
- Try matching eval-zone GR against known-zone GR (self-correlation) instead of typewell
- Multi-scale offset search (different σ values per well based on eval zone length)
- Neural sequence models that can learn the GR→TVT mapping from cross-well training data
- Image-based features from the .png cross-sections
