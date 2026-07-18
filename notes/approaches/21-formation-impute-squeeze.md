# 21 — Squeeze the spatial lever: better formation-surface imputation — 2026-07-18

The formation surfaces are a NEAR-PERFECT TVT signal (`formation_plane_probe`: ORACLE own-tops → eval
RMSE **0.006**). Everything between that and our score is IMPUTATION ERROR of the surfaces from neighbour
wells. So the spatial lever's headroom is enormous IF imputation improves. Squeeze = better imputation.

## Honest validation harness (NOT grouped CV)
Grouped CV strips pad-neighbours → understates spatial features 20× (the project's hard lesson). The right
proxy is **leave-one-well-out with the neighbour cloud RETAINED** (self-excluded), which matches the test
scenario (scored wells are offset wells to the full train set). `tools/formation_impute_variants.py` uses
this and reports mean-of-6 tvtF RMSE **bucketed by nearest-neighbour distance** — because the value lives in
the near buckets (test wells have close neighbours; on train only ~6% of rows do).

## Result: universal kriging @ k=60 (SHIPPED)
Compared linear-plane / quadratic / IDW / universal-kriging (plane trend + IDW of plane residuals) × k.
`<200ft` near-set mean-of-6 tvtF RMSE (ffill 14.73, oracle 0.005):

| variant | near-set RMSE |
|---|---|
| `plane_k12` (**old production**) | 59.94 |
| `uk_k40` | 30.83 |
| **`uk_k60`** (**shipped**) | **30.75** |
| `uk_k80` | 30.75 (saturates) |
| `twin_uk40` (nearest-twin override) | 33.65 (worse — dropped) |
| `quad_k24/40` | 147–302 (overfits sparse neighbours) |

End-to-end through the PRODUCTION `_FormationPlaneKNN` (raw-coord plane + UK residual): **pooled** tvtF RMSE
**88.7 → 25.2**. The k=10 plane was wildly extrapolating on the 45% isolated (500+ft) wells (123 RMSE); UK
+ k=60 replaces that with a stable regional surface (~24). Better across ALL distance buckets, not just near.

**Why it works:** (1) k=10 fits a plane to 10 noisy local well-medians → unstable slope → huge toe
extrapolation error; k=60 averages a regional trend that's far more stable. (2) Universal kriging adds an
IDW correction of the plane residuals, recovering local structure the global trend misses, AND is robust to
trend mis-fit (the residual absorbs it) — so raw-coordinate numerical wobble doesn't hurt. Twin-override
hurts because a single well's MEDIAN depth doesn't match the query point's local depth; the fitted surface
is better even for close twins. Quadratic overfits the sparse neighbour set catastrophically.

## Change shipped (src/model.py + kernel, identical)
`_FormationPlaneKNN`: default `k=10 → 60`; `impute()` adds the UK residual term
`resid_q = IDW(fn − plane(xn,yn))` to the plane prediction. Same 22-feature interface, same `knn_d` gates,
same test-legality (only X,Y + neighbours' surfaces). Per-well impute ~6× slower (bigger neighbour fetch)
but still a small fraction of the ~1.4h kernel.

## Validation status
Grouped/stack CV UNDERSTATES this (near-set barely appears when pad-neighbours are stripped) → the ONLY
trustworthy arbiter is the **public LB** (every spatial gain in this project was validated there, not on CV).
NEXT: push kernel, submit, compare vs v20/11.020. If it helps → keep; if flat/worse → revert (the imputation
metric is much better, but the GBM-interaction / knn_d-gate effect is what public decides).

## Open (untried, higher headroom toward oracle 0.005)
POINT-LEVEL imputation: fit through nearby formation PICKS (not well-medians), preserving along-lateral
structure. Well-median loses that each lateral spans a large area at varying depth. Candidate next squeeze
if uk_k60 confirms on public.
