# Approach 11 — Typewell Geology: a LEAKAGE trap (REVERTED)

**Status:** 🚨 REVERTED — the feature is TRAIN-ONLY; the gain was leakage that cannot transfer.
**Date:** 2026-07-13

## 0. THE VERDICT (read first)
The typewell `Geology` column exists **only in train** — test typewells have just `TVT, GR`
(verified: all 3 example test typewells lack the column, and the Kaggle run confirmed it by
scoring *worse*). So `_geology_features` falls back to zeros at inference: the model was trained
on real geology values but sees constant zeros in test → **public 13.601 → 13.927 (worse)**.
The −0.53 CV / −0.62 lockbox gain was real *on train wells* but is pure leakage — CV and lockbox
both used train wells (which have Geology), so neither could detect the feature's absence in test;
only the real-test public LB exposed it. **Lesson: verify a feature EXISTS at inference (check
`data/raw/test`) before building on it.** Diagnostics that caught it: `tools/critique_diag.py`
(bootstrap CI confirmed the gain was "real" on train; the test-well representativeness check showed
the geo features fall back to zeros — the smoking gun). The sections below are the (now-invalid)
original analysis, kept as a record of the trap.

---

## [ORIGINAL, INVALID] Typewell Geology "cracks" the LEVEL component

**Status:** concluded-kept (current best; biggest single gain in the project)
**Date:** 2026-07-13
**Config / commit:** `configs/gbm_stack.yaml` (predictor `gbm_stack`, now incl. geology features)

## 1. Idea
notes/09 concluded the per-well LEVEL was geology-locked after 4 levers died at offset R²≈0.10.
But those levers were all **GR cross-correlation** (spatial NN, global/band-limited GR alignment)
— they alias on the quasi-periodic GR. We never tested the typewell **`Geology` column**: a
categorical formation label vs TVT. A horizontal lateral *holds inside a target formation*, whose
absolute TVT the typewell gives us — and formations are **ordered in TVT, not periodic**, so
matching to a formation is immune to the aliasing that killed GR alignment. And `Geology` is the
*one geological signal available at test time* (formation-top columns and `.png` are train-only).

## 2. Gate — the oracle then the test-legal version (`tools/geology_probe.py`)
Target = offset (eval_mean TVT − last_known).
| predictor | offset R² | corr |
|----|----|----|
| baseline (5-feat anchor) | 0.098 | — |
| baseline + **last-known's formation** (test-legal) | **0.187** | +0.348 |
| baseline + oracle (true) formation | 0.245 | +0.443 |

The corr +0.35–0.44 is 4–5× anything prior (spatial 0.09, GR-align 0.17). Mechanism: if last-known
sits near the bottom of the target bed, the well drifts *up* into it — so the formation's TVT
relative to last-known predicts the drift. All test-legal (uses known last TVT + typewell `Geology`).

## 3. In the full stack — the decider (`tools/geology_stack.py`)
Three per-well features added to the LGB+Cat @ step=5 stack:
`geo_last_off` (last-known formation median − last), `geo_hmm_off` (HMM-aligned eval formation
median − last), `geo_last_med` (that formation's absolute median TVT).

| metric | current stack | + geology | Δ |
|----|----|----|----|
| **pooled 773-well CV** | 13.4892 | **12.9982** | **−0.49** |
| **lockbox** (154 held-out) | 13.5050 | **12.8858** | **−0.62** |

Both metrics agree, and **pooled CV has no selection bias** (one predefined feature set, not
best-of-N) — so this is real, unlike the phase-2 lockbox read. The signal is orthogonal to the
HMM (which uses GR *alignment*, not formation *labels*), so it adds on top of the stack's existing
0.337 level recovery. Canonical `reproduce_cv`: 13.520 → (see log).

## 4. Conclusion
**The level component was NOT fully closed — I had tested GR channels but not the formation-label
channel.** The typewell `Geology` column is the strongest lever since the HMM: stack CV ~13.5 →
~13.0. Corrects notes/09's "level is geology-locked" — level is partly recoverable *through the
formation labels*, which are the one piece of geology that survives into the test set. Remaining
room: a GR→formation classifier for the eval zone could push the test-legal 0.187 toward the
oracle 0.245. See [[level-ceiling]] (now amended), [[error-budget]], notes/09, notes/10.
