# 18 — Prefix-CV per-well self-calibration — NEGATIVE (2026-07-18)

The 2nd queued lever after TabICL. Honest per-well adaptivity: hold out the last 25% of each well's KNOWN
prefix, gauge how reliable the well's drift is there, shrink the eval prediction toward the safe anchor
where unreliable. Validated LOCALLY on the cached OOF before any kernel work:
- per-well prefix-drift-error vs per-well eval-error: **corr −0.01** (no signal — the probe error, dominated
  by the curve section's nonlinearity, doesn't identify unreliable wells).
- EVERY shrink-toward-anchor variant (thr∈{5,8,12}, k∈{0.3,0.6,1.0}): pooled 11.37–13.10 vs base 10.90 — all
  WORSE. The drift IS signal; shrinking toward the flat anchor loses it.
**Confirms the standing concern:** per-well adaptivity has now hurt 4× (adaptive blend, fusion cut, TabICL
tune, prefix-CV). The faithful top-team version (regenerate candidate trajectories on the shortened prefix,
select per well) is expensive AND its candidates overlap our GBM features; one top author calls prefix-CV
"a public-LB-only trick that doesn't transfer to private." Not shipped. tools/prefix_cv_probe.py.

**BEST STANDS: kernel v20, public 11.020** (TabICL). Levers now exhausted: spatial formation-plane (worked),
MLP (worked), lik-PF (marginal), TabICL (worked, −0.022) vs GR (closed ×3), kriging (neg), meta/blend/
prefix-CV per-well adaptivity (all CV-good/public-bad or negative).
