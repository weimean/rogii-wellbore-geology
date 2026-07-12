# Approach 07 — Particle filter as a diverse stack feature

**Status:** concluded-kept (current best, 15% over ffill)
**Date:** 2026-07-12
**Config / commit:** `configs/gbm_stack.yaml` (predictor `gbm_stack`, now incl. PF features)

## 1. Idea & motivation
After four consecutive negatives (self-corr redundant, resolution hurts, LightGBM ties,
HMM tuning overfits), the one genuinely *different* tracker we had not built was a particle
filter. amerhu reports PF+HMM blends beat either alone — diversity, not raw accuracy, is the
value. A PF is a different *algorithm* on the same typewell reference, so it makes different
errors than the grid HMM.

## 2. Method
`pf_posterior` — bootstrap particle filter, state = (TVT position, dip-rate):
- 600 particles; init pos ~ N(last_tvt, 3), rate ~ N(tail_ir, 0.01).
- Predict: OU rate (mom=0.98, sig_r) + position drift rate·ΔMD (sig_p); clip to band.
- Update: Student-t emission on (GR − typewell_GR(pos))/gs; systematic resample when ESS<N/2.
- Forward-only (no smoothing) — deliberately different from the HMM's forward-backward.
Its posterior mean+std are added to the GBM stack as features: `pf_pred, pf_std, pf−anchor,
hmm−pf, pf_fusion`. Final blend bumped to 0.6·(anchor+GBM) + 0.4·hmm_fusion.

## 3. Validation — full 773-well GroupKFold (global RMSE)
| Model | RMSE | Δ vs ffill |
|----|----|----|
| ffill | 15.910 | — |
| hmm_fusion | 14.216 | −10.7% |
| gbm_stack (no PF) | 13.769 | −13.4% |
| **gbm_stack + PF** | **13.530** | **−15.0%** |
Reproduced by `tools/reproduce_cv.py` (honest OOF anchor via `anchor_override`).

## 4. Why it works when the self-reference HMM didn't
| | self-ref HMM | particle filter |
|--|--|--|
| reference | own pre-PS GR (different) | typewell (same as HMM) |
| algorithm | same grid FB HMM | **different** (MC forward + resample) |
| result | redundant (+0.001) | **complementary (+0.24)** |
Diversity comes from a *different algorithm making different errors*, not a different
reference. Raw PF (19.4) and pf_fusion (15.2) are both worse than the HMM alone — but that's
irrelevant; the GBM extracts the complementary signal.

## 5. Uncertainty
- PF adds MC stochasticity (seed=0 fixed); 600 particles keep it stable.
- Blend 0.6 and the PF params mildly tuned; landscape flat. Public-LB read pending.
- Compute: PF is cheap (~35s for all 773 wells).

## 6. Conclusion
The particle filter broke a four-negative plateau: 13.77 → 13.53. The lesson generalizes —
**ensemble diversity should come from different algorithms, not restatements of the same one.**

## 7. Public-LB divergence + where we stopped (2026-07-13)
Submitting gbm_stack+PF (kernel v4): **public 13.854 — WORSE than v3's no-PF 13.727**, even
though CV improved 13.77→13.53. The public LB is 3 leaked, non-representative wells (a scored
subset); ±0.13 there is noise (same divergence seen for hmm_fusion: CV 14.24 / public 15.12).
**We trust the 773-well OOF CV** (it estimates the private ~200-well LB) and keep the PF version.

A follow-up self-reference PF (different algo AND different reference) added only +0.07
(13.530→13.459) — below the public-LB noise floor. Given the divergence, chasing sub-0.1 CV
gains is poor risk/reward, so **self-PF was NOT productionized** and we locked in gbm_stack+PF.
The tracker-diversity vein is saturated; further LB gains need the frontier's full pipeline.
