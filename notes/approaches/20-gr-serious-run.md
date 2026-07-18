# 20 — Serious GR run: ~5 is honestly achievable (leaderboard-proven) — 2026-07-18

Leaderboard reality check corrected our framing: top HONEST teams reach ~5 (shu01 4.859, Tucker 5.44,
Rishikesh 5.24, k256 5.86). We're 11.020 (rank 3214/5173, below median). So there IS a reproducible honest
lever we never found — the top teams' GR/feature representation beats everything we + 2 agents tried.

## What FAILED before (don't repeat)
- Direct GR prediction (datum scan, affine-cal, bimodal, DTW, self-corr): capped ~12 (point-matching oracle).
- Windowed-NCC refinement of the HMM path (18 feats): −0.016 (redundant).
- Our rich features are POINT-wise residuals (hgr − tw_gr(anchor+off)) — a weak per-row signal.

## The NEW direction (untried properly): rich GR-alignment FEATURES → GBDT
k256 (12th, ~5.86): "tabular LGB+Cat+XGB, the INPUT FEATURES matter." Tucker (7th, 5.44): single model,
per-well only. So the signal is a FEATURE FAMILY that answers "what TVT does THIS GR window match", fed to a
GBDT — not a direct predictor. Build the FULL family (lesson: never conclude from per-feature probes):
- NCC-peak-TVT: per eval row, slide a window of horizontal GR (sizes 15/31/61/121) against typewell GR(TVT);
  emit the TVT of the peak NCC, the peak value, the 2nd-peak TVT (bimodality), peak sharpness. Multi-window.
- Multi-smoothing: repeat at GR σ = 1/3/8 (separate coarse structure from fine).
- Self-reference: same NCC-peak-TVT against the KNOWN-zone horizontal GR (no calibration, same tool).
- Cross-signals: NCC-peak-TVT vs anchor / HMM / PF.
Measure grouped-CV OOF delta over base+rich+formation (caches exist). Ship if it adds.
