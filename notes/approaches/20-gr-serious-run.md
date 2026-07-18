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

## RESULT (2026-07-18) — NEGATIVE, the GR lever is now fully closed
Built the full family: **128 NCC-peak-TVT features × 3.78M rows** (`/tmp/rogii_grpeak.npz`,
`tools/gr_peak_probe.py build`) — typewell (`gp_tw_*`) and self (`gp_self_*`) NCC-peak TVT-offset, peak
value, 2nd-peak, sharpness, across smoothing σ={1,3,8} × windows {8,15,30,60}, plus aggregate stats
(`gpx_*` offstd/pkmax/refdiff/tw_self_off). Measured grouped-CV OOF (LGB+Cat seed-ens, w=0.75, step=5,
`tools/grpeak_stack_test.py`):

```
base (rich + formation):   10.9167
+ 128 NCC-peak-TVT feats:  10.9197   delta −0.0030  (WORSE)
```

**Flat-to-negative. The GR lever is exhausted, including this last untried representation.** Grouped CV is
TRUSTWORTHY here (per-row self+typewell GR, NO neighbour wells → no 20× understatement caveat, unlike the
spatial family). Why it failed despite being the "build the whole family, don't probe one at a time" method
that broke the false 13.52 ceiling: (1) the peak-LOCATION signal is too noisy — GR self-similarity (corr
0.65 at true TVT) means the NCC peak routinely lands on the wrong offset, so `off1/pk1` are near-random per
row; (2) whatever weak signal exists is REDUNDANT with the rich GR-match residual grids (`tda/tdpf/tdhmm`)
already in the base — the GBDT already extracts it. **The rich-GR-match family worked (−0.27) because it
encodes the residual VALUES the model can regress; the peak-TVT family re-encodes the same physics as a
noisier argmax and adds nothing.**

**CONCLUSION: the top teams' honest ~5 is NOT reachable via any GR-alignment FEATURE representation we can
build (three now tried: direct-predict, NCC-refinement, full peak-TVT family — all flat/negative).** It is
their PROPRIETARY human/AI-guided dip interpretation (the ROGII StarSteer workflow, notes/16), which is not a
reproducible offline feature. BEST STAYS v20/public 11.020. The GR route is closed for good; do not re-open
it without a genuinely different representation (none identified).
