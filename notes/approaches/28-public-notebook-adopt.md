# 28 — Learning from the public notebooks: the honest ceiling is ~7.5, not a leak — 2026-07-29

**Trigger (user):** competitors uploaded public notebooks/code — study them and adopt what's honest.
Pulled 6 kernels via the API and read them in parallel. Field has moved: public LB top is now **4.68**
(was ~7–11), the public "7159" family scores ~7.16, and even our 10.70 is mid-pack. Competition is
**still live (deadline 2026-08-05)**, so this reopened active work.

## The headline correction — our README was wrong
The top public architecture ("rogii-7159-safe-twin-router", 5 votes; and the identical `public-v2`
forks) accounts for its own score in its results table:

| stage | LB | transfers to private? |
|---|---|---|
| last-known baseline | ~15.9 | — |
| **A⊕B honest dual-pipeline blend** | **~7.5–7.6** | **YES** |
| + guarded well-ID contact override | ~7.16 headline | NO (no-op on hidden test) |
| + Gold visible-prefix overlay (OFF by default) | ~7.2–7.3 | NO |

**The leak is worth only ~0.3–0.4 and is a no-op on the private LB.** The honest machinery alone reaches
~7.5 — beating our 10.70 by ~3 RMSE. So the old thesis ([[project_leaderboard_intel]], and the README's
"gap to top is a non-transferable leak, deliberately not chased") is **falsified**: the bulk of the gap is
honest method we never built. (The 7.5 is self-reported/unverified for the rebuild, but three independent
notebooks share the architecture and the whole field sits at 4–6, so it's credible.) The `pipea-only`
kernel = Engine A alone, leak-free (spatial imputers self-exclude the query well); v14 (public **139**) is a
naive ID-copy with no physics — nothing to take. The two `public-v2` kernels are byte-identical forks.

## Where their honest edge lives: physics candidate generators + post-processing, NOT the GBDT
Their booster stack is the same LGB+Cat family we already found ties HistGBM ([[project_model_vs_features]]).
The edge is signals we don't have:

- **U = TVT+Z robust-poly projection** (`project_well_A`) — "the move that stabilises drift."
- **Warm-up damping** — ramp predictions out of the last known TVT (`d *= 1−exp(−md/τ)`, τ≈85).
- **Likelihood-weighted multi-seed PF** — 128 seeds × 4 temps, softmax by *whole-trajectory* GR log-lik.
- **Beam-search GR aligner** + **multi-scale NCC** — different alignment *algorithms* → decorrelated members.
- **`seg_b_well` segment offsets** (full/early/mid/late thirds + robust WLS) — models the drifting per-well
  offset (our error budget is [[project_error_budget]] ~68% LEVEL).
- **GR-residual-at-candidate-offset features** (`hgr − tw_gr(path ± {−5,0,5})`).
- Safe-override pattern (if ever revisited): MD-interpolated, prefix-verified (RMSE≤1ft, ≥50 rows,
  provably ≥ blend) — far better than the naive twin-copy we refuted.

## ✅ ADOPTED (Stage 1) — post-processors, kernel v29
**U=TVT+Z projection + warm-up damping.** The formation surface U=TVT+Z is smooth & ~polynomial over the
eval zone while TVT wiggles with the trajectory; robust Tukey-IRLS deg-4 fit to U vs sequence, then
TVT = U_smooth − Z re-injects the exact trajectory. Config = **deg 4, blend 0.75, τ=100** (matches the
public `project_well_A` 75/25). Grouped-OOF (`tools/`-style probe on `rogii_meta_oof.npz`, baseline 10.8775):

| step | OOF | Δ |
|---|---|---|
| baseline blend | 10.8775 | — |
| + U=TVT+Z projection (deg4, blend0.75) | 10.7147 | −0.163 |
| + warm-up damping (τ=100) | **10.7043** | **−0.173** |

Sign matters: `tvt+z` −0.16, `tvt`-alone +0.30, `tvt-z` +2.3 (confirms the public convention). Robust, not
noise-fit: **76% of wells improve** (586/773), median per-well −0.14, both well-halves consistent
(−0.171 / −0.157), flat optimum plateau over deg 3–4 / blend 0.6–0.8. This is a per-well deterministic
transform in the geological coordinate → same category as the sequence-model lever that grouped CV
**understated 5×** (notes/27), NOT a blend knob → expect public gain ≥ −0.17. Wired into
`kernel/rogii_hmm_fusion.py` and `src/model.py` (`_postproc_well`). **Public LB result: PENDING (v29).**

## Stage 2 — queued
New decorrelated candidate generators (beam-search aligner, multi-scale NCC, likelihood-weighted PF) +
`seg_b_well`/GR-residual features. Heavy PF ensembles to run on Kaggle compute.
