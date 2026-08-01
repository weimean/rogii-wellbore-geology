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
**understated 5×** (notes/27), NOT a blend knob. Wired into `kernel/rogii_hmm_fusion.py` and `src/model.py`
(`_postproc_well`). **✅ PUBLIC LB: v29 = 10.565 (ref 55087026), from v27 10.699 = −0.134.** Transferred
(OOF −0.173 → public −0.134; slightly less because the test base uses full-train models = less drift to
correct). NEW BEST. Confirms the "adopt public honest levers" thesis.

## Stage 2 audit (2026-07-29) — we already hold most of the levers
Grep of kernel/src: likelihood-weighted multi-seed PF (`pf_lik_ensemble`, notes/14), `seg_b_well` segment
offsets, and GR-residual-at-candidate-offset features are ALREADY present. The two candidate generators we
genuinely DON'T have: **beam-search DP aligner** and **multi-scale NCC self-correlation** (own known-zone GR
template vs eval windows → a TVT path; distinct from our spatial DenseANCC). Since we hold most levers yet
sit ~3 RMSE behind ~7.5, part of the gap is likely tracker QUALITY + their physics-dominant blend (0.7
physics vs our 0.25), not missing features — probe that alongside the two new generators.

## Stage 2 — mostly NULL (2026-07-29), the new generators hit our GR-weakness wall
Validated on cached OOF (`scratchpad/stage2_*.py`, beam aligned to cache by truth-block matching):

| lever | result | verdict |
|---|---|---|
| **beam-search DP aligner** (banded Viterbi, L1 GR-mismatch + move penalty) | alone **43.05** RMSE; blend w=0.05 **+0.39** | NULL — raw GR has no level anchor, drifts to wrong TVT branches (GR SNR≈0.02, [[gr-signal-weakness]]). Residual-corr vs base 0.38, not even decorrelated. |
| **physics-upweight** (their 0.7-physics arch) | fusion 0.25→0.40→0.70 monotonically WORSE; fusion alone 13.83 vs anchor+corr **10.68** | REFUTED — our fusion is a *weak* member (theirs is strong); we already correctly down-weight it. Their architecture doesn't transfer. |
| **fusion-downweight** 0.25→0.10 | OOF −0.087 BUT half A −0.173 / half B **flat** | MIRAGE — blend-knob overfit to one subset (cf. 4 historical fusion-weight public reverts; grouped CV overstates blend knobs). Not submittable. |
| **multi-scale NCC self-corr** | not built | Same "recover TVT by GR alignment" family as beam; its raw-GR-shape signal already tested NULL in notes/27 ("last breath"). Building = re-confirming that null. |

**Conclusion:** we already hold the public honest *feature* levers (likpf/seg_b/GR-resid, adopted notes/14+);
the genuinely-new GR-alignment generators all fail on our weak GR. The one transferable public lever was the
U=TVT+Z projection → **v29 10.565 (shipped)**.

## Is the claimed ~7.5 even real? VERIFIED BY REPRODUCTION (2026-07-29) — NO.
Ran the one fully-reproducible public honest pipeline (LB52 `adoredwhale` = PF + 14-config beam + selector +
deg-5 robust projection; no pretrained artifacts) on OUR 200 labeled train wells (guarded: only TVT_input
fed in, eval rows scored vs true TVT, pooled RMSE). `scratchpad/verify/`:

| component | pooled RMSE (200 wells) |
|---|---|
| PF base (their workhorse) | 101.96 |
| full selector (shipped output) | 79.98 |
| + deg-5 projection | 80.02 |
| beam alone (best honest component) | 16.09 |

**Nothing near 7.5.** The PF DIVERGES — GR likelihood (SNR≈0.02, [[gr-signal-weakness]]) can't anchor the
momentum random-walk, so it drifts 80–160 while true TVT stays in a ~15–40 band. PF output was bit-identical
at 16/64/128 seeds → not a seed-count issue. The shipped **"7.159" is a bundled precomputed CSV**
(`public_score_7_159_submission.csv`, loaded in preference to running the physics) — an opaque artifact, NOT
physics output. So the twin-router's self-reported "honest ~7.5" table is UNVERIFIED marketing; the LB top
(~4.68) is leak/artifact-driven on PUBLIC. **This corrects my mid-session over-correction** (I'd rewritten
the record to claim a real ~7.5 honest ceiling based on their markdown tables — WRONG). The ORIGINAL thesis
was closer: the gap to the top is leak/artifact, not reproducible honest method. **10.565 is near the honest
reproducible ceiling.** LESSON: verify a claimed ceiling by REPRODUCING it, not by trusting a self-report.

## Round 2 (2026-07-30) — four more untried avenues, all CLOSED
User pushed back that avenues remained. Fair — round 1 only closed candidate *generators* (beam/NCC).
Tested four more via a fast single-LGB grouped-OOF harness (`scratchpad/fast_ablate.py`, baseline postproc
10.7632) for the feature routes, and an LB submission for the TCN route:

| avenue | route | result | verdict |
|---|---|---|---|
| A — spatial→dip prior | GBM feature | −0.005 OOF (noise); raw WORSE | NULL. corr(spatial_dip, true_dip)=0.48 is real, but a row-independent GBM can't use a per-row dip-rate (it already has the spatial surface via tvtF). |
| B — global structural surface | GBM feature | +0.003 OOF | NULL. Thin-plate-spline LOWO surface helps raw blend (−0.008) but the U-projection already captures that smoothing → nets to zero. |
| D — GR affine calibration | GBM feature | +0.023 OOF | NEGATIVE. Extra a/b/residual cols let the GBM overfit. |
| **C — spatial→TCN** | LB (v30) | **10.722 (+0.157)** | REVERTED. Feeding spatial+formation channels (_TCN_DROP 30→5) overfits the data-starved 773-well TCN — exactly the notes/27 "more channels hurt" lesson. |

**Not pursued:** A's *tracker* route (feed spatial dip into HMM/PF sequentially) — the trackers are a weak
0.25 member we already downweight, and the U-projection already stabilises the drift the dip would target,
so EV is low; whole-pipeline bagging (~−0.02 safe, not worth a submission slot alone). The corr-0.48 spatial
dip integrates to 81 (diverges) — better than local-dip (110) but nowhere near usable standalone.

**FINAL: three feature routes null/negative + the TCN route regressed. This round every avenue was tested
empirically, not assumed. v29 / public 10.565 stands as the best; 10.565 is the honest reproducible ceiling.**

## Round 3 (2026-07-31) — user asked "what have we not tried"; 5 more avenues, 1 WIN
Pushed past the "we're done" call again. Five avenues, each pre-checked cheaply before any rebuild:

| # | avenue | result | verdict |
|---|---|---|---|
| 1 | prefix bias+scale calibration | corr(known-slope, eval-slope)=0.04 | NEGATIVE. The per-well eval trend is NOT estimable from the known zone (known resid-slope std 0.073 vs eval 0.005, different regime); offset already captured by the Ridge anchor. Oracle-linear 5.87 needs eval truth we don't have. |
| 3 | Huber loss (δ=10) on LGB+Cat | OOF −0.064 → **LB v31 10.596 (+0.031)** | REVERTED. Single-LGB OOF gain concentrated in one well-half; washed out / hurt in the full stack. Lesson: single-LGB OOF over-reads a loss change. |
| **2** | **spatial-dip PF as a FEATURE** | **OOF −0.148, both halves (A −0.233, B −0.061), 12× seed noise** | **WIN (v32).** The tracker ROUTE that round 2 dismissed. A PF whose rate prior reverts toward the neighbour-well structural dip (`pf_spatial`, spw=0.5); decorrelated from local-dip trackers. NB the raw dip-rate FEATURE was null (round 2 A) — the PF *prediction* integrating the dip is what carries. Fusion/HMM route (dip_center) REGRESSED (+0.092) and was dropped. |
| 4 | spatial neighbour definition | not run | queued |
| 5 | whole-pipeline bagging | not run | queued |

**Key reversal:** round 2 pre-judged A's tracker route as low-EV and skipped it. Round 3's cheap PF pre-check
(spatial dip −2.29 on raw PF output, spw plateau 0.5–0.7) showed the tracker DOES use the dip well; the
payoff came through the PF *as a decorrelated feature*, not the variance-gated HMM fusion. **v32 = v29 +
spatial-dip PF feature (Huber dropped).**

**✅ v32 PUBLIC LB = 10.424 (ref 55166231), from v29 10.565 = −0.141. NEW BEST.** OOF −0.148 → public
−0.141: clean ~1:1 transfer (a decorrelated FEATURE tracks CV, unlike the blend knobs). Progression:
10.699 → 10.565 → **10.424**. #2 was the strongest lever of the whole public-notebook effort, and it's the
tracker route round 2 dismissed. Remaining: #4 (neighbour definition), #5 (bagging), now measured vs 10.424.
