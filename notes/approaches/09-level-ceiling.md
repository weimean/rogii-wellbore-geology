# Approach 09 — The LEVEL ceiling (why ~9 is not reachable)

**Status:** concluded-negative (definitive; do not re-drill LEVEL)
**Date:** 2026-07-13
**Tools:** `tools/level_probe.py`, `tools/level_headroom.py`, `tools/level_align.py`

## 1. Reframe
"Climb to ~9" is a LEVEL problem, not a wiggle/model problem. Competition RMSE decomposes as
`RMSE² ≈ 9.04² (oracle-constant wiggle floor) + level-RMSE²`, and
`level-RMSE = offset_std(12.69)·√(1−offset R²)`. So:

| offset R² | level-RMSE | competition CV |
|----|----|----|
| 0.337 (now) | 10.33 | 13.54 |
| 0.50 | 8.97 | ~12.7 |
| 0.70 | 6.95 | ~11.4 |
| 0.95 | 2.84 | ~9.5 |

Reaching ~9 requires recovering **~95% of the per-well offset variance**.

## 2. Where the stack already is
`level_headroom.py` — the full stack (HMM+PF+GBM+A/B) already recovers **offset R²=0.337**
(level-RMSE 10.33), far above the 5-feat anchor's 0.098. It works by variance-weighting the
alignment (trust it only where hmm_std is low). Notably the **HMM's own eval-mean is worse than
ffill** (offset R²=−0.078): good wiggle, drifting level.

## 3. Four independent LEVEL levers — all dead at the ~0.10 wall
| Lever | result |
|----|----|
| velocity+GR anchor (Ridge, 5 feats) | offset R² 0.098 |
| spatial K-NN (offset of nearest wells by X,Y) | corr 0.094, R² < 0 |
| global typewell GR alignment (fit level+slope by NCC) | corr 0.084, R² −4.6 |
| band-limited anchor-seeded alignment | corr 0.168, but only +0.007 over anchor |

Wells *are* spatially interspersed (median PS–PS 478 ft) but the offset has no spatial
structure. The global alignment fails to **quasi-periodic GR aliasing** — it finds corr≈0.46 at
the *wrong* TVT cycle. Constraining the band doubles the raw corr (0.084→0.168), confirming
aliasing, but the refined signal is redundant with the anchor that seeded it.

## 4. Verdict
The per-well level is at its **input-limited ceiling**. The missing ~66% of offset variance is
geology (bed dip/azimuth) not present in the test-available inputs (X, Y, Z, GR, typewell;
formation tops and .png are train-only). **~9 is not reachable on the real distribution**, and
even 0.5 offset R² isn't. `gbm_stack+PF+A+B` at 13.54 is at/near the genuine ceiling. The public
~9 is 3 leaked wells, consistent with this. See [[error-budget]], [[oracle-bounds]], notes/03.
