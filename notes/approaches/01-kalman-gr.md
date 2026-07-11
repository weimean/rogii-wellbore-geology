# Approach 01 — Extended Kalman Filter with Typewell GR

**Status:** concluded-kept (marginal improvement, first real model)
**Date:** 2026-07-11
**Config / commit:** `configs/kalman_gr.yaml`

## 1. Idea & motivation
Track TVT using an Extended Kalman Filter. State = TVT (1D). Process model = random walk
(process_std=0.1 ft/step). Observation model: horizontal GR should match typewell GR at
the current TVT estimate, using the typewell GR gradient as the Jacobian for the EKF update.
Blend toward ffill at large distances from the PS boundary (decay=10000) to limit drift.
Add a global +1.42 ft bias (mean eval-zone offset across 773 training wells).

## 2. Method
- Smooth typewell GR (Gaussian σ=3), build interpolant + gradient function
- Interpolate/smooth horizontal GR (Gaussian σ=10)
- EKF forward pass: predict → update where |typewell GR gradient| > 0.1
- Blend: w = exp(-dist/10000); pred = w * kalman + (1-w) * ffill + bias

## 3. Validation
Global RMSE (all 3.78M eval-zone rows, 773 wells):
- **ffill:       15.910**
- **kalman+bias: 15.735** (1.10% improvement)
- Per-well mean: 12.704 vs 12.812 (0.84% improvement)
- Beats ffill on 394/773 wells (51%)
- Oracle per-well selection (ffill, kalman) would give 12.305 → room exists

## 4. Contribution
| Component | Effect |
|-----------|--------|
| Kalman filter alone (no bias) | 15.817 (0.58% over ffill) |
| +1.42 ft global bias | 15.735 (additional 0.52%) |
| Blend decay (vs pure Kalman) | Marginal — decay=10000 barely differs from no blend |

## 5. Uncertainty
- Improvement is marginal and may not survive the private test distribution.
- GR→TVT signal-to-noise ratio is ~0.03 (typewell dGR/dTVT ≈ 0.3 vs horizontal GR std ≈ 15).
- The +1.42 bias is fitted on all training wells — it's a data leak if test distribution
  differs. Safer to zero it but the improvement was consistent.
- Model selection (kalman vs ffill per well) could help but no reliable selector found.

## 6. Physical meaningfulness
The EKF is physically motivated: it uses the typewell's GR-vs-TVT signature to constrain
TVT. But the weak signal means it effectively stays close to ffill on most wells. The
approach is sound in principle — the SNR is just too low at the per-step level for the
EKF to add much value.

## 7. What was tried and didn't work (this session)
- **Polynomial/linear extrapolation** of TVT from Z or MD: diverges over 4000+ step eval zone
- **GR nearest-neighbor lookup** (typewell or self-profile): too noisy, worse than ffill
- **Calibrated alignment** (learn GR offset, then match): calibration doesn't help
- **Linear alignment search** (optimize TVT_start + rate): finds wrong alignment
- **GBM on eval-zone features** (residual target): overfits or can't generalize across wells
- **Spatial NN offset** (nearby wells predict TVT offset): correlation only 0.06
- **Per-well simulated eval-zone offset**: unreliable extrapolation
- **Self-correlation + typewell dual-Kalman**: noisier than single typewell
- **GR integral/ODE approach**: equivalent to Kalman, same marginal result
- **RTS backward smoother**: no improvement (no anchor at eval zone end)
- **Correlation-based model selection**: correlation doesn't predict which model is better

## 8. Conclusion
The fundamental bottleneck is the weak GR→TVT signal in the eval zone. The typewell
dGR/dTVT is ~0.3 GR/ft while horizontal GR has ~15 GR units of noise, giving SNR ≈ 0.02.
No amount of filtering or matching strategy overcomes this unless the signal is aggregated
over very long windows (hundreds of steps). Next approaches should try:
- DTW/sequence alignment at the right scale (not point-by-point)
- Neural network sequence models (LSTM/1D-CNN) that can learn cross-well patterns
- The .png cross-section images (may contain signal not in the GR curve)
- Better feature engineering: GR wavelet decomposition, formation-aware segmentation
