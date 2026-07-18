# 23 — Output postprocessing (per-well Savitzky-Golay + distance ramp) — 2026-07-19

The sub-9 notebooks all shape the OUTPUT: per-well Savitzky-Golay smoothing (win 17, poly 3), a distance
ramp `(1−e^{−md_since/τ})` on the correction, and a delta-from-last-known target. Since we're at tracker
parity (notes/22), output-shaping was the untested honest lever. Tested on the CACHED OOF (no retrain,
`tools/postproc_probe.py`), base 10.9034:

| transform | best OOF | delta |
|---|---|---|
| per-well Savitzky-Golay (win 7..51) | 10.9022 (win=51) | −0.0012 (negligible) |
| distance ramp (τ=20..500) | 10.9036 → 10.9738 | ALL WORSE |
| savgol + ramp | worse | — |

**NEGATIVE / non-lever for us.** Root cause: **our predictions are already smooth** — soft posterior means
(HMM forward-backward + PF) + an anchor, not the sub-9 teams' JAGGED hard beam-DP paths. Savitzky-Golay has
almost nothing to remove (−0.001, below the public noise floor; bigger windows "help" only by over-smoothing,
which risks real structure on the true test). The distance ramp hurts because our `anchor + correction`
structure already weights the correction optimally across the eval zone (we tuned exactly this — the blend
0.75 and the anchor are the same idea). Delta-target: we already predict `truth − anchor` (anchor is pinned
to the last-known level), so we already have the delta framing.

**Corollary of tracker parity:** the postprocessing that helps the sub-9 teams is fixing a jaggedness we
don't have. NOT shipped. BEST STAYS v20/11.020.

## Session scorecard (three cheap honest levers, all closed)
- Spatial imputation squeeze (uk_k60): better metric, WORSE public (knn_d gate) — notes/21.
- Beam-DP dip family: redundant with our HMM (corr 0.9997) — notes/22.
- Output postprocessing: non-lever, our predictions already smooth — notes/23.

**Only genuinely-orthogonal candidate left untested:** z-velocity-coupled PF (`dTVT/dMD = β·dZ/dMD + c`) —
a GEOMETRY prior from trajectory Z rather than GR, so unlike the beam it might decorrelate from our
GR-driven trackers. Everything else points to the honest ceiling being ~11.0 for us; the sub-9 gap is the
train/test well-ID leak (non-transferable to private) + proprietary interpretation we can't reproduce.
