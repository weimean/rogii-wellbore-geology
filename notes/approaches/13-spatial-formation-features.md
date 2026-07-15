# 13 — Spatial formation features (the structural-datum lever)

**Idea → validation → conclusion.** After the rich-features + MLP ceiling break (notes/12, public 12.948),
the next top-team family: use the TRAIN-only formation surfaces, imputed spatially from neighbour wells,
to pin the per-well datum drift. Chosen deliberately by the user over TabICL / stopping.

## 1. The structural fact (why this could work)
The horizontal CSV has 6 formation columns `ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA` — these are **per-row
structural-surface depths** (the depth of each geological marker at the well's current X,Y along the lateral;
they drift as the well crosses dipping geology). Empirically:

- `D = TVT + Z` (the "datum function") has within-well std **62 ft** — that's the level+trend we fail to model.
- `D ~ 6 formation surfaces` (per-well linear fit): **R² = 1.0000 in 100% of wells.**
- **Oracle** (true surfaces in eval, offset calibrated on pre-PS): eval RMSE **0.01**. TVT is *deterministic*
  in the surfaces. (The old "oracle tops +0.014" probe used SCALAR per-well tops — wrong object; the per-ROW
  surfaces that encode lateral dip are the signal.)

So if we could impute the surfaces at the eval X,Y, the problem is solved.

## 2. Test-legality (the geology-leak lesson applied)
Formation columns are **TRAIN-only** (test horizontal = `MD,X,Y,Z,GR,TVT_input`). But unlike the geology
leak, the surfaces are globally-consistent absolute depths, so we build a **reference cloud from train wells**
`(X,Y) → mean surface` and impute at a *test* well's own X,Y (KNN, self-excluded). This never touches the
test well's absent formation columns — it's the same operation at train (LOWO) and inference. Test-legal.

## 3. The catch — imputation accuracy, and it's neighbour-gated
- KNN imputation error ≈ **18–19 ft** (median nn-dist to another well = 356 ft). The eval-zone drift is only
  ~8–12 ft (Z is ~flat in the horizontal section), so for the typical **isolated** well the imputation noise
  *swamps* the signal → loses to ffill.
- **But** for wells with a genuine close neighbour (<50 ft, same pad): ffill 16.0 → spatial **10.5** (+5.6).
  Only **4.6%** of eval rows have such a neighbour (96% of train wells are geographically isolated).

## 4. Result — proven, small, safe
Built LOWO-honest features aligned to the stack (`_cache_spatial.py`), measured incremental over the rich stack:

| stack (LGB+Cat, w=0.75) | pooled CV |
|----|----|
| rich features | 13.2007 |
| **+ spatial (sp_drift, sp_nndist, sp_impd)** | **13.1514** |
| Δ | **+0.049** |

Small (only 4.6% of wells benefit) but real, above noise, and **orthogonal** to GR-matching (structural, not GR).
Refutes the old "spatial-tops fail on our CV" note — it wasn't that the signal is absent, it's that close
neighbours are rare.

## 5. Why it's SAFE (unlike geology)
The GBM keys on `sp_nndist` (nearest other-well distance). A well with no close train neighbour → large nndist →
the spatial prediction falls back to the base model. Worst case on a geographically held-out test set is **+0.00
(no harm)**; best case +0.05. Contrast geology, which fell back to a *wrong constant* and hurt public 13.6→13.9.

## 6. Production
`build_spatial_ref` + `_spatial_cols` in `src/model.py` (+ kernel mirror), threaded through
`_stack_features / fit_gbm_stack / gbm_stack`. Production `_spatial_cols` verified to match the validated
`_cache_spatial` output **exactly** (max diff 0.0). Submit smoke-test clean (14,151 rows, NaN=0). Kernel v14.

## 7. PUBLIC RESULT: 11.839 (−1.11) — the scored test wells ARE offset wells
v14 public = **11.839** (ref 54717721), from 12.948. The grouped CV (+0.05) understated it **20×**: grouping
by well removes each well's pad-neighbours, so on train only 4.6% of wells have a close reference neighbour;
the scored test wells sit near train wells, so the feature fires on far more of them (oracle 0.01 where it
fires). Legitimate offset-well geosteering, not a target leak. Biggest single jump in the project. Opens the
path toward top-team ~7-11 — the crude mean-surface drift already gave −1.11; stronger imputation (neighbour
TVT directly, all 6 surfaces separately, full-eval reconstruction toward the 0.01 oracle) should push much further.

See [[diversity-and-ceiling]], notes/12.
