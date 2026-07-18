# 24 — z-velocity-coupled PF (geometry prior) — 2026-07-19

The last genuinely-orthogonal candidate (notes/22-23): a GEOMETRY prior `dTVT/dMD = β·dZ/dMD + c`, fit per
well on the known prefix, extrapolated over the eval zone using the KNOWN eval Z path. The deterministic
mean of the sub-9 teams' `run_pf_z`. Distinct from all our GR-driven trackers — IF Z predicts TVT.

## RESULT: catastrophic. The premise is geologically false. CLOSED.
`tools/z_coupled_probe.py`, 773 wells, 3.78M eval rows (ffill 15.910, anchor 15.193):

| predictor | pooled RMSE |
|---|---|
| z-coupled `dTVT=β·dZ` | **108.3** |
| z-coupled `dTVT=β·dZ+c·dMD` | **52.5** |
| free MD-trend `dTVT=c·dMD` | 1330.9 |

All catastrophic; blending with the anchor only makes it worse monotonically. **`dTVT` is NOT predicted by
`dZ`** (β·dZ = RMSE 108 ≫ ffill 15.9), and the per-well fit extrapolated over 74% of the well accumulates
drift and explodes.

**Why (geology):** in a horizontal well the bit stays roughly in-zone (small Z wiggle); TVT changes are
driven by the STRUCTURAL DIP of the formation, not the bit's vertical motion. So `dZ` carries almost no
information about `dTVT`. Confirms the pre-existing "datum U=T+Z catastrophic" and "slope R²≈0" findings.

**Why the sub-9 `run_pf_z` isn't a counterexample:** it's not a pure geometry predictor — it's a GR
particle filter with a z-informed MOTION prior; the GR likelihood does the work. Strip GR → collapses
(this probe). Keep GR → redundant with our HMM/PF (dip-rate motion + GR emission), like the beam-DP
(notes/22). Either way, no orthogonal signal.

## Session close-out: honest ceiling = 11.020
Four candidates tested this session, all closed:
1. Spatial imputation squeeze (uk_k60): better metric, WORSE public (knn_d gate) — notes/21.
2. Beam-DP dip family: redundant with our HMM (corr 0.9997) — notes/22.
3. Output postprocessing (savgol+ramp): non-lever, predictions already smooth — notes/23.
4. z-coupled PF: geometry premise geologically false (dZ ⊥ dTVT) — notes/24.

**We are at parity with the top teams on everything reproducible** (tracking, features, output shape). The
remaining gap to sub-9 is the train/test well-ID leak (non-transferable to private — we don't chase it) +
proprietary human interpretation that isn't an offline feature. BEST = v20 / public 11.020, fully honest.
