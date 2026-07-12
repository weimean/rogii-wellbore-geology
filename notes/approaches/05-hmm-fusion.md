# Approach 05 — 2nd-order HMM ⊕ level_regression fusion

**Status:** concluded-kept (current best, 10.5% over ffill)
**Date:** 2026-07-12
**Config / commit:** `configs/hmm_fusion.yaml`

## 1. Idea & motivation
Studying the top public notebooks (via `kaggle kernels pull`) showed the wiggle IS
recoverable — our earlier "unpredictable" call was a limitation of our 1st-order Kalman,
not the data. The winning family is a **2nd-order (TVT, dip-rate) sequential tracker**.
We built our own — informed by, not copied from, amerhu's "Exact HMM Smoother" — and
**fused it with our `level_regression`** (our distinct edge on the LEVEL component).

## 2. Method
**2nd-order forward-backward HMM** (`hmm_posterior`):
- Joint state = (TVT position on a 0.5-ft grid, dip-rate). Grid spans last_known ± 90 ft
  (clipped to typewell range). 21 dip-rate values, span ±0.035.
- Transition: **OU mean-reverting dip-rate** (mom=0.98, sig_r=0.003) + position drifts by
  rate×ΔMD (sig_p=0.45). The OU velocity is the physics our variance-vs-lag EDA found
  (bounded mean-reverting drift) — we set it from that, not from LB tuning.
- Emission: **Student-t** (df=4, robust to GR outliers) on (GR − calibrated typewell GR)/gs,
  gs = clipped std of the known-zone GR-vs-typewell residual.
- Grid HMM ⇒ represents **multi-modal** posteriors, so the full-sequence forward-backward
  smoother resolves the quasi-periodic **GR aliasing** that broke our NCC/Kalman.
- Eval GR sub-sampled to 250 bins for speed (GR is ~60-120× oversampled vs TVT); posterior
  mean + variance interpolated back to all rows.

**Variance-aware fusion** (`hmm_fusion`): w_hmm = 1/(1+(std/blend_k)²), blend_k=4;
pred = w_hmm·HMM + (1−w_hmm)·level_regression. Trust the HMM where GR is informative (low
posterior variance), fall back to the level anchor where it drifts.

## 3. Validation — full 773-well GroupKFold (global pooled RMSE)
HMM is per-well (no cross-well leakage) → computed directly. `level_regression` anchor is
OOF (GroupKFold). Fusion per-well.
| Model | RMSE | Δ vs ffill |
|----|----|----|
| ffill | 15.910 | — |
| level_regression | 15.325 | −3.7% |
| HMM raw | 16.469 | +3.5% (drifts alone) |
| HMM ⊕ ffill (var blend) | 14.602 | −8.2% |
| **HMM ⊕ level (var blend, k=4)** | **14.239** | **−10.5%** |

~3× the improvement of level_regression. The raw HMM alone is *worse* than ffill — the
variance blend is what makes it win (catches the drift, falls back to the anchor).

## 4. What is ours vs theirs
- **Theirs (principles):** 2nd-order (TVT, dip-rate) state, mean-reverting velocity, grid
  HMM for multimodality, forward-backward smoothing, Student-t/tempered emission.
- **Ours (novel):** (a) fusing the tracker with a separately-validated **level model** by
  posterior variance — the winners lean on the tracker alone; we combine our level edge
  with their wiggle edge; (b) OU parameters set from our **own EDA** (mean-reversion
  horizon), not tuned on the contaminated public LB → more private-LB-robust.

## 5. Uncertainty
- blend_k=4 tuned on CV (mild overfit risk; the RMSE-vs-k curve is flat 4–6).
- HMM assumes test wells share the geosteering dynamics of train (reasonable).
- Compute: ~0.4 s/well (250 bins) — well within the 9 h kernel budget for ~200 wells.

## 6. Physical meaningfulness
High. The state (TVT, dip-rate) with a mean-reverting rate is literal geosteering: the bit
drifts at some inclination-vs-dip rate that decays back toward bed-parallel. The variance
fusion encodes "believe the GR correlation only when it's sharp."

## 7. Conclusion
Reframing the wiggle as a 2nd-order sequential-tracking problem — and fusing it with our
level model — nearly triples our margin over ffill (3.7% → 10.5%). Next levers: multi-scale
/ self-correlation emission (own pre-PS GR profile, per slide 9), a prefix-CV harness to
tune constants without touching the real eval zones, and tuning band/step/S for the tail.
