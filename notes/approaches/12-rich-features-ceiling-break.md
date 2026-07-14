# Approach 12 — Reproduced top-team features BREAK the "ceiling"

**Status:** concluded-kept (biggest gain since the HMM; overturns the ceiling conclusion)
**Date:** 2026-07-14/15
**Config / commit:** `configs/gbm_stack.yaml` (predictor `gbm_stack`, now incl. rich features)

## 1. The turn
After declaring an "input-limited ceiling at 13.5" (notes/09-11, domain-confirmed), we inspected the
top teams' artifacts (`fleongg/rogii-claude-models-pub`, `thbdh5765/rogii-v10-fresh-artifacts`). They
are **legit trained models** (LGB+Cat+TabICL, not a raw answer leak) with **honest train OOF ~11.1**
(vs our 13.47). Their edge = **~200 features** (vs our 37) + **TabICL** (tabular foundation model,
their best single base at 11.28). Their public 7.016 has an extra ~4-pt boost beyond honest OOF
(leak-flavored, not reproducible); but **~11 IS honestly reproducible.**

## 2. What we reproduced (the cheap, test-legal half)
`_rich_cols` (src/model.py + kernel; `_cache_richfeat.py`): 54 features, all from horizontal GR +
typewell GR/TVT + our HMM/PF predictors:
- **Multi-offset GR-match grids** — `hgr - typewell_GR(predictor + offset)` around last_tvt (`tda`,
  offsets ±80), PF (`tdpf`), HMM (`tdhmm`) at ±[2..30]. A GR-match *profile* around each predictor,
  telling the GBM whether the GR fits better slightly above/below the prediction. THE key novel family.
- Finer multi-scale GR roll mean/std (`grm/grs` @5/21/51/101), lag/lead (`glag/glead` @1/5/15/30),
  `gr_d1/d2/env/nrg`, `gr_vs_tw_anc`.

## 3. Result (both metrics agree)
| | pooled CV | lockbox |
|----|----|----|
| base (37 feats) | 13.473 | 13.295 |
| **+ rich (91 feats)** | **13.201** | **12.997** |
| Δ | **−0.272** | **−0.298** |

Biggest single gain since the HMM fusion. Verified: submit runs clean on test wells (rich features
are test-legal, no leakage).

## 4. The lesson (why we missed it)
We tested these ideas **in isolation** — spatial "dead" (corr 0.09), tops oracle "+0.014" — and
declared a ceiling. **In aggregate, fed to a GBM, they clear ~0.3+ (and ~2 with the full set + TabICL).**
Never conclude "ceiling" from per-feature probes; test the full rich feature set + a strong model.

## 5. Remaining runway (toward their ~11.8 GBM / ~11.1 ensemble)
- Per-formation references (`tvtF_*` × 6 formations) + spatial/"dense" imputation **as features**
  (not the direct reconstruction we tried and failed).
- **TabICL** (GPU; needs a token or koolbox-offline install). Their best single model.
See [[level-ceiling]] (corrected), notes/09-11.

## 6. Neural-net (MLP) diversity — the second lever (2026-07-15)
"Have we tried a neural net?" — no. Added a plain sklearn `MLPRegressor((256,128,64), relu, alpha=1e-3,
early_stopping)` on the same 91-feature matrix (StandardScaler, 200k-subsample train for speed). No torch,
so it runs on the Kaggle CPU/GPU image.

| | pooled CV (w=0.75) | note |
|----|----|----|
| LGB+Cat (seed-ens) | 13.201 | rich features |
| MLP alone | 14.652 | worse alone — expected |
| **0.7·GBM + 0.3·MLP** | **13.004** | −0.20; corr(GBM,MLP)=0.64 |

Same principle as the particle filter: diversity from a **different function class** helps even when the
new model is worse alone. Lockbox OVERSTATED the gain (−0.58 → 12.41); the pooled 773-well read (−0.20)
is the honest one. Productionized in `src/model.py::fit_gbm_stack`/`gbm_stack` + kernel mirror; submit
smoke-test clean (14,151 rows, NaN=0). Public read: kernel v13, pending daily-limit reset.

**Discipline note:** two CV-good/public-bad episodes (geology leak, adaptive blend) mean the public LB is
the arbiter. Confirm v13's public score before stacking more speculative features on this −0.47 gain.
