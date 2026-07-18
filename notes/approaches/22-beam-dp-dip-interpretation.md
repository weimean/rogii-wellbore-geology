# 22 — Beam-DP dip interpretation (the top-team "auto-dip") — 2026-07-19

## What it is (the answer to "what is the proprietary dip interpretation")
Research (parallel agent) nailed it: the top teams' "superior GR interpretation" is a **constrained
beam-search Viterbi / DTW** over a TVT-vs-MD trellis. States = typewell TVT-grid indices; stages =
smoothed horizontal-GR samples along MD; cost = `(GR_h − GR_tw(TVT))²/emit + move·|ΔTVT|`, dip bounded to
±2 cells/step, beam-pruned. Run at ~7 stiffness settings → a FAMILY of TVT paths, fed as GBM features (NOT
the direct prediction). This is the constrained cousin of the per-point NCC-peak (which was noise) — the
`move` continuity penalty resolves GR's many-to-one ambiguity by picking the smoothest globally-consistent
path. Faithfully reproduced in `tools/beam_dp_probe.py` (numpy beam core, 7 stiffness regimes).

## Go/no-go: PASSED (unlike NCC-peak)
Direct predictor sanity (250 wells): ffill 16.088, our HMM 16.057, **beam mean-of-7 15.993** — the family
beats ffill AND our HMM standalone. So the paths are legitimate tracker outputs (not noise), and the
stiffness family beats any single path. Went ahead to the stack test.

## Stack OOF: NEGATIVE −0.074, REDUNDANT with our HMM. NOT shipped.
Built the full family cache (10 feats × 3.78M rows: 7 paths + mean + std + GR-misfit) and ran the honest
grouped-CV stack OOF on top of base (rich+formation):

```
base (rich + formation):   10.9167
+ beam-DP family (10):     10.9911   delta −0.0744  (WORSE)
```

Grouped CV is **trustworthy** here — the beam uses only each well's OWN GR + typewell (no neighbour wells),
so the spatial-understatement caveat does NOT apply (same as the NCC-peak features). So −0.074 is real.

**Root cause — REDUNDANCY (confirmed):** `corr(beam_mean, hmm_pred) = 0.9997`. The beam family-mean is
99.97% identical to our existing 2nd-order (TVT, dip-rate) forward-backward HMM. The 7 paths wander ±10.6 ft
around the HMM path (the stiffness sweep) but average right back to it; `beam_std` (disagreement) is tiny
(~1.5 ft). So the beam-DP is a different SOLVER (hard Viterbi) of the SAME GR→typewell problem our HMM
already solves (soft forward-backward) — a restatement, not a new signal. Its 10 columns add no orthogonal
information and give the GBM extra surface to overfit → −0.074.

**This is the project's core principle, reconfirmed:** *diversity must come from a DIFFERENT ALGORITHM, not
a restatement* (self-ref HMM was redundant +0.001; PF — a genuinely different tracker — added −0.24). The
beam-DP is the top teams' tracker, but WE ALREADY HAVE AN EQUIVALENT ONE (2nd-order HMM + 2 PFs + lik-PF).
For teams without our HMM/PF set, the beam WAS their tracker and mattered; for us it's a restatement.

## What this means for the sub-9 gap
The top-team edge is NOT the beam-DP dip tracker per se — we are at parity on tracking (corr 0.9997). Their
sub-9 comes from some combination of: (a) SPATIAL datum recovery done better (but our spatial squeeze just
BACKFIRED — notes/21 — the knn_d gate over-trusts a smoothed surface), (b) postprocessing (delta-target +
per-well Savitzky-Golay + distance ramp), and/or (c) the train/test well-ID leak (non-transferable to
private). Tracking is a solved, saturated lever for us.

## Remaining untried candidates from the research (NOT yet tested)
- **z-velocity-coupled PF** (`dTVT/dMD = β·dZ/dMD + intercept`): a GEOMETRY prior using trajectory Z, not GR
  — potentially orthogonal to all our GR-driven trackers (unlike the beam). The most promising remaining item.
- Postprocessing: delta-target + distance ramp `(1−e^{−md_since/τ})` + per-well Savitzky-Golay smoothing.
- XGBoost as a 3rd GBM (function-class diversity, per our own principle).
- Full rows instead of step-5 subsampling (with stronger regularization).
