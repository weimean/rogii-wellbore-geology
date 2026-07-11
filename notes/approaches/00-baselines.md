# Approach 00 — trivial baselines (floors)

**Status:** concluded-kept (as floors, not contenders)
**Date:** 2026-07-11
**Config / commit:** `configs/baseline.yaml`

## 1. Idea & motivation
Establish an honest, submittable floor and prove the pipeline (load → predict → submit)
runs end-to-end before any real modelling.

## 2. Method
- `baseline_ffill`: carry the last known `TVT_input` forward into the gap.
- `baseline_linear_interp`: interpolate `TVT` across the gap by `MD`, then bfill/ffill.

## 3. Validation
Not yet scored on CV (no held-out TVT used); verified structurally instead — the generated
submission's id set matches `sample_submission.csv` exactly (14,151 rows, 0 NaN preds).

## 5. Uncertainty
Because the eval zone is the well **tail**, there is no known TVT *after* the gap, so
`linear_interp` degenerates to a constant last-known value (~11747 for well `000d7d20`).
These are flat floors, reliable nowhere in particular.

## 6. Physical meaningfulness
None — they encode "geology stays flat past PS", which is exactly the assumption the task
exists to break.

## 7. Conclusion
**Keep as floors to beat.** Confirms the real signal must come from GR-signature matching
(self-correlation → typewell → offset-well dip), not gap interpolation. Next: Approach 01,
self-correlation.
