# 14 — Top-team deep dive (what actually wins) — 2026-07-15

Pulled + analysed the current top public notebooks (9.251 DTW, 7.295 rebuild, 7.201, 7.159, 7.091,
sunnywu physical, dual-pipelines). **All clean top solutions are the same recipe**, and its core lever is a
RICHER version of our spatial-formation win.

## The universal honest lever: spatial offset-well formation-PLANE datum recovery
`FormationPlaneKNN`: per train well store (median X, median Y, median depth of each of 6 formations
ANCC/ASTNU/ASTNL/EGFDU/EGFDL/BUDA). For a target well, find the k≈10 nearest OTHER wells (KDTree on median
XY, self-excluded) and fit a local **WLS plane** `depth ≈ a·X + b·Y + c` (IDW-weighted); evaluate at the
target's eval (X,Y) → the formation surface along the lateral (regional dip). Then:
```
TVT_eval = −Z_eval + formation_surface(X,Y) + b_well
b_well   = median(TVT_known + Z_known − formation_surface)   over the visible prefix
```
Shape/dip from neighbours; per-well datum pinned by the known prefix. Needs only X,Y at inference (no
train-only columns on the test well). Plus `DenseANCCImputer` (dense IDW k=15-20 over all wells' ANCC-vs-XY).
Features: tvtF_{6}, tvt_dense, b_{full/early/mid/late/wls}, spatial_knn_dist, pf_vs_spatial, etc.

**Why our crude version only got grouped-CV +0.05 but theirs gets huge:** we did a KNN AVERAGE of the
mean-surface (needs a CLOSE neighbour → only 4.6% of wells). The PLANE FIT extracts the regional dip from
k=10 spread-out neighbours → works for MOST wells (formation surfaces are laterally smooth over km). So the
plane version should show a LARGE grouped-CV gain, unlike ours.

## Their honest CV ladder (their ffill 15.91 == ours → directly comparable)
ffill 15.91 → plain LGBM (delta target) 10.85 → + spatial + lik-PF feats 9.69 → full stack **9.21**.
Our current best = **11.84 public** (crude spatial). ~2.5 ft of honest, transferable headroom.

## Supporting cast (all honest)
delta target (TVT−last_known); trackers AS FEATURES (128-seed likelihood-weighted PF × scales{3,5,8,12},
14-config beam, multi-scale NCC); GR-offset grids (we have these); robust **IRLS deg-4 projection of U=TVT+Z**
vs normalized MD (75/25); **two independent stacks blended 0.55/0.45**; warmup damping (τ≈85); per-well
Savitzky-Golay smoothing.

## The ~7 public scores = a LEAK (authors confess it)
Train/test EXACT well-ID overlap: `if wid in train_wells: reconstruct answer from the TRAIN copy`
(`tvt_from_contacts` using train-only Geology/formation cols + true TVT), guarded by prefix-RMSE<1ft. Takes
~9.2 CV → ~7.2 public. The 7.159 author's own markdown: "a public-LB-only trick… does not transfer to
private." NOTE: our earlier "no exploitable leak" probe matched by GR-similarity, NOT exact well-ID — so the
exact-ID channel is UNTESTED by us. Likely a public mirage; private-LB is where it evaporates.

## Plan
Build the rich FormationPlaneKNN family (this session). Skip the leak (public-only, non-transferable).
See [[winning-methods]], [[spatial-offset-wells]], [[level-ceiling]], notes/13.
