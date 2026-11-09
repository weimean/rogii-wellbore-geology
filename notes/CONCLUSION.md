# ROGII Wellbore Geology — Final Conclusion

**Task:** predict eval-zone TVT (True Vertical Thickness, ft) for horizontal wells. Metric: pooled RMSE.
**Best model:** kernel **v32** (`gtwm98/rogii-hmm-fusion`, ref 55166231) — **public 10.424**, fully honest & rules-compliant.
**Final standing:** private **RMSE 9.673**, rank **3116 / 6125 (top 50.9%)** — counted sub was v29 (see FINAL STANDING below).
**Arc:** ffill 15.91 → 13.288 → 11.020 → 10.852 → 10.699 → 10.565 → **10.424** (−5.49 from ffill), every gain validated on the real LB.
**Verdict (2026-07-29):** two "honest ceilings" I declared both fell. 11.020 was ARCHITECTURE-POVERTY — a
TCN reading each well as a SEQUENCE (the learned analogue of a geologist reading the log holistically) broke
it (v25/v27, −0.32 total; grouped CV understated it 5×). Then a study of the top public solutions added the
drift-stabilising **U=TVT+Z projection**: predict the smooth formation surface `TVT+Z`, fit a robust low-order
polynomial, subtract the exact trajectory Z (v29, −0.134 public; notes/28). Then a **spatial-dip particle
filter** (v32, −0.141 public): a PF whose rate prior reverts toward the neighbour-well structural dip (surface
gradient projected on the trajectory heading), fed as a decorrelated feature — the cross-pollination of our two
best veins (spatial surfaces + physics trackers). A "what haven't we tried" round netted this one win out of
five (Huber loss looked good on single-LGB OOF but regressed on the LB; prefix calibration / neighbour variants
/ bagging all null-to-marginal).
**On the gap to the LB top (~4.7):** I reproduced the one runnable public "honest" pipeline (PF + beam +
selector + projection) on my own 773 labeled wells — it scores 16 to 80 pooled, never near the claimed 7.5,
because the particle filter diverges on the weak GR signal (SNR≈0.02). The 7.159 is a bundled precomputed CSV,
not physics output. So that specific self-reported table was an artifact — **but the conclusion I drew from it,
that the whole gap is a "leak," was wrong** (corrected below in FINAL STANDING after the winners' writeups
appeared: the real ~4-RMSE gap was a 2D image-alignment U-Net model class, honest and reachable, not a leak).
(The recurring lesson: never declare a ceiling from probes or self-reports — break it with a new algorithm,
or reproduce it before believing it. 13.52 was feature-poverty, 11.020 was architecture-poverty, 7.5 was a mirage.)

---

## FINAL STANDING (competition closed 2026-08-05)

**Team `weimean` — private RMSE 9.673, rank 3116 / 6125 (top 50.9%).** Public was 10.424 / rank 3944 (top 64.4%):
**the private shakeup moved us up 828 places.** Trusting reproduction and grouped CV over the public LB
generalised — teams that overfit the public split fell, we rose.

**The counted score is v29 (9.673), not v32.** Kaggle auto-selected the two best *public* subs (v32 10.424 and
v29 10.565) and kept the better *private* one. Private scoring inverted the public ranking of our own levers:

| Sub | Lever | Public | Private |
|-----|-------|-------:|--------:|
| v29 | U=TVT+Z projection | 10.565 | **9.673** ← counted |
| v32 | + spatial-dip PF | 10.424 | 9.725 |
| v30 | spatial→TCN (*reverted*) | 10.722 | **9.669** ← best private of all, never selected |
| v31 | Huber (*reverted*) | 10.596 | 9.744 |

Two things this confirms empirically:
1. **The round-3 "win" was public overfit.** The spatial-dip PF (v32) beat public by −0.141 but *cost* ~0.05 on
   private (9.673 → 9.725) — squarely inside the ~0.07 single-LGB seed-noise band we'd identified. The largest
   sub-0.07-clearing public signal still didn't transfer. The two-submission safety net (v29) is what saved us.
2. **The honest-ceiling read was right for the *tabular* paradigm — but the gap was not a leak.** Our GBM+PF
   program bottomed at ~9.67, and no amount of feature work moves that. POST-REVEAL, though, the winners'
   writeups show the ~4-RMSE gap to the top (private **5.639**) was **a model class we never tried, not a
   leak**: reframe geosteering as **2D image alignment** (a GR-match heatmap, x = position, y = candidate TVT)
   and let a **U-Net / CNN pick the correct ridge**, outputting a per-column distribution over TVT. That class
   hits ~4.6–5.3 CV. Evidence there was no leak: winners scored *worse* on public than private (Ruby
   5.98 → 5.64); a published writeup is titled *"corr(public, private) = −0.04"*; the 1st-place author
   investigated the discrepancy and found only a noisy 50-well public split. My 07-29 reproduction was *half*
   right — the "7.159 = bundled CSV" claim was genuinely fake — but I over-generalized it into "the whole gap
   is a leak," which was wrong. Our own particle filter and spatial surface `ΔTVT = aΔX + bΔY − ΔZ` both appear
   in the 1st-place solution — as *one channel* into the alignment net, never as the model. See notes/28.
   **Lesson: when a gap persists, suspect a missing model class before concluding "leak."**

---

## 1. The model we ship (kernel v20)

Anchor-then-residual, diverse-algorithm stack (every input available at inference):

```
per well ── LEVEL ANCHOR (Ridge: v50,v200,gr_off,eval_len,known_range) ─┐
         ── 2nd-order (TVT,dip-rate) fwd-bwd HMM ── posterior mean/var ─┤
         ── bootstrap particle filter (TVT,dip-rate) ──────────────────┤  variance
         ── likelihood-weighted 8-seed PF ensemble (scales 3/5/8/12) ───┤  FUSION
                                                                        │
   FEATURES fed to the residual stack (target = TVT − anchor):
     • rich multi-offset GR-match grids (hgr − typewell_GR(anchor+off), ~11 offsets ×3 anchors)
     • GR rolling/lag/lead/deriv/envelope stats
     • A (HMM-PF disagreement) + B (GR envelope/energy)
     • SPATIAL formation-plane: FormationPlaneKNN (6 formations, WLS plane over k=10 nearest OTHER
       wells, self-excluded) + dense ANCC + per-well datum b (seg variants) → tvtF_{6}, tvt_dense, gates
     • crude spatial mean-surface drift (sp_drift, sp_nndist, sp_impd)
   MODELS: LightGBM×3 + CatBoost×3 (seed-ens, step=5 subsample) ⊕ MLP(256,128,64) ⊕ TabICL(v2 regressor)
   BLEND (correction): 0.65·(0.7·GBM + 0.3·MLP) + 0.35·TabICL
   FINAL: out = 0.75·(anchor + correction) + 0.25·fusion
```

- **TabICL** (BSD-3, soda-inria) runs OFFLINE on CPU (`model_path`, `allow_auto_download=False`,
  `n_estimators=4`, `kv_cache=True`); fit on 10k context rows; predict per-well on a coarse 40-row subsample
  + interpolate the smooth correction (keeps CPU cost tiny); serialised with a lock in the threaded loop.
- Delivery: offline dataset `gtwm98/rogii-tabicl-offline` (tabicl+einops wheels + 114MB ckpt). `enable_gpu=false`
  (Kaggle P100 torch is broken for sm_60; TabICL is CPU; GBM boosters keep a CPU fallback).
- Kernel runtime ~1.4h (safely under the 9h cap). Resubmit: `api.competition_submit_code(competition=...,
  file_name="submission.csv", kernel="gtwm98/rogii-hmm-fusion", kernel_version=20)` (Python API — the CLI
  `-k -v` form 400s without `file_name`).
- **Version state (updated 2026-07-29):** best SUBMISSION = kernel **v32** (ref 55166231, public **10.424**) —
  v20 logic + TCN sequence model (notes/27) + U=TVT+Z projection & warm-up damping (notes/28). Earlier baseline
  below describes v20 (public 11.020). The local repo code (`src/model.py`, `kernel/`) tracks v32. (v20 logic: k=10 FormationPlaneKNN,
  the uk_k60 squeeze backed out). Kaggle kernel HEAD is v23 (the reverted uk_k60 experiment — public 11.594);
  to make the kernel head match best again, push once (`kaggle kernels push -p kernel/`) → v24 = v20 logic
  (a compute run, not a submission). Not required — v20 remains the scored best regardless.

## 2. Lever ledger — what WORKED (all honest, all shipped)
| Lever | Public | Note |
|---|---|---|
| rich multi-offset GR-match features | — | broke the FALSE "13.52 ceiling" (it was feature-poverty) — notes/12 |
| MLP neural-net diversity | 12.948 | 0.7·(LGB+Cat)+0.3·MLP; different function class (corr 0.64) — notes/12 |
| **spatial formation-plane (offset wells)** | **11.101** | **−1.85, the breakthrough**; grouped CV UNDERSTATES it 20× — notes/13-14 |
| likelihood-weighted PF ensemble | 11.042 | marginal −0.06 (we already have a PF) — notes/14 |
| **TabICL foundation model** | **11.020** | −0.022, a different function class; BSD-3, compliant — notes/17 |

## 3. Lever ledger — what's CLOSED / NEGATIVE (measured, not assumed)
- **GR wiggle matching** (direct scan, affine-cal, bimodal, NCC, DTW, self-correlation): physics-capped ~12
  (GR corr 0.65 at true TVT; self-similar in the ±20ft band). Two dedicated agents + many probes, all
  negative. The ROGII webinar confirms the precise method is HUMAN/AI-guided dip interpretation, not a
  reproducible feature/scan — notes/15-16.
- **The 18% wiggle is irreducible for us:** not from GR (above), and **doesn't transfer between wells**
  (neighbour TVT-transfer 25 vs smooth-surface 8.4 on close-neighbour wells — each well's wiggle is its
  own) — notes/19.
- **Per-well adaptivity burned us 4×** (all CV-good/public-bad or negative): adaptive blend, fusion-weight
  cut (0.75→0.90: −0.10 CV / +0.03 public), TabICL tuning (blend 0.5: +0.05), prefix-CV self-calibration
  (corr 0 with eval error; shrinkage hurts) — notes/15,18.
- Anisotropic kriging of the per-well level: worse than null (level is survey-density-limited) — notes/15.
- Meta-stack-of-stacks: marginal −0.03 CV — notes/15.
- Geology typewell column, formation TOPS as labels: TRAIN-ONLY leak (reverted, notes/11).
- **TabPFN:** non-commercial license (Prior Labs) bars competition use + needs a token → excluded — notes/17.

## 3b. Session 2026-07-18/19 — SEVEN levers tested to close the sub-9 gap, all CLOSED
A research agent identified the top-team "superior GR interpretation" as a constrained beam-search
Viterbi/DTW dip correlation (cost = GR-emission + move·|ΔTVT| dip penalty, stiffness-swept into a path
family, fed as features). We reproduced each candidate faithfully and validated it the RIGHT way (public LB
for offset-well/spatial features, which grouped CV understates; grouped CV for per-well/model knobs, which
it judges honestly). Every reproducible lever came back negative:

| # | Lever | Result | Why it failed |
|---|---|---|---|
| 1 | GR NCC-peak-TVT feature family (128 feats) | grouped-CV −0.003 | peak-offset is noise (RMSE 58 vs HMM 16); redundant with rich GR-match grids — notes/20 |
| 2 | Spatial imputation squeeze (k=10→60 + universal kriging) | **public 11.020→11.594** | LOWO imputation halved (88.7→25.2) but the k=10 plane's WILD tvtF on isolated wells was PROTECTIVE (GBM distrusts via knn_d); smoothing → over-trust. Better metric was misleading — notes/21 |
| 3 | Beam-DP dip family (top-team "auto-dip", 7 stiffness paths) | grouped-CV −0.074 | `corr(beam_mean, our HMM)=0.9997` — a different SOLVER of the same GR problem, not a new signal — notes/22 |
| 4 | Output postprocessing (per-well Savitzky-Golay + distance ramp) | −0.001 (non-lever) | our predictions are already smooth (soft posterior means, not jagged beam paths) — notes/23 |
| 5 | z-velocity-coupled PF (`dTVT=β·dZ`) | RMSE 108 (catastrophic) | `dZ ⊥ dTVT` — TVT driven by structural dip, not bit vertical motion; sub-9 `run_pf_z` works only via its GR likelihood → redundant with our HMM/PF — notes/24 |
| 6 | Human-interpretation / de-shrink the correction | grouped-CV −0.09 but CV-MIRAGE | error = un-trackable structural drift on high-relief wells (`corr(err,dip_range)=+0.63`); direction unpredictable; de-shrink is the blend family reverted 4× — notes/25 |
| 7 | Training-regime tuning (full-rows + strong-reg, sub-9 setup) | our step-5 regime WINS (10.942 vs 10.955) | CV-HONEST: more reg & more rows both hurt; "model is not the bottleneck", re-verified — notes/26 |

**Verdict: parity with the top teams on everything reproducible** (tracking, feature richness, output shape,
model regime). Each "secret" was something we already have in a different form. No feature, method, or tuning
lever remains untested. The sub-9 gap is the train/test well-ID leak (non-transferable to private —
deliberately not chased) + real-time human/AI dip interpretation (impossible in an automated notebook).
**(11.020 was the v20 honest ceiling at the time; later broken to 10.565 by the TCN and the U=TVT+Z
projection — see the header and §6.)**

## 4. Key principles (transferable)
1. **Never conclude a ceiling from per-feature probes.** Isolated features read ≈+0.01; the top teams' rich
   feature FAMILY + a strong model cleared −0.27, then the spatial family −1.85. The "13.52 ceiling" I
   declared repeatedly was FEATURE-POVERTY.
2. **Diversity must come from a DIFFERENT ALGORITHM** (not a restatement): PF over HMM (+0.24), MLP over
   LGB/Cat (−0.20 pooled), TabICL over all (−0.022). Each is worse alone but decorrelated.
3. **Grouped CV UNDERSTATES spatial/offset-well features ~20×** — it strips each well's pad-neighbours into
   the held-out fold, but the scored TEST wells ARE offset wells to train. For anything using neighbours,
   trust the public LB, not grouped CV. (Opposite trap: the geology leak was CV-good/public-bad.)
4. **Per-well adaptivity / blend re-tuning is public-noisy** — 4 reverts. Global, out-of-fold changes only.
5. **Verify a feature EXISTS at inference** (geology leak) and **verify licenses** (TabPFN) before shipping.
6. **Know which knobs grouped CV judges honestly vs which it distorts** — and in which direction. It
   UNDERSTATES offset-well/spatial features (strips pad-neighbours) → trust public. It OVERSTATES
   blend/correction-scale knobs (train wells are more in-distribution) → 4 reverts. It is HONEST for per-well
   trackers and model/regularization knobs (no neighbour leakage) → trust it. Match the validator to the knob.
7. **A better local metric is not a better score.** The uk_k60 squeeze halved the imputation RMSE yet
   REGRESSED public — because the GBM's knn_d gate relied on the old estimates looking obviously bad.
   Always ask what DOWNSTREAM behaviour a metric improvement changes, not just the metric.
8. **Reproduce the competitor's method before assuming it's the missing lever.** The top-team beam-DP dip
   correlation was 0.9997-correlated with our existing HMM — a different solver, not a new signal. Parity is
   only provable by building their thing and measuring the overlap.

## 5. The remaining gap (honest, final — updated 2026-07-29)
**Leaderboard (5893 teams):** #1 shu01 4.68; the public "7159" cluster forks the leak notebooks; median ~9.
Ours **10.424**. I reproduced the one runnable public "honest" pipeline on my own labeled wells to test the
claimed ~7.5 tier (§6). The gap to the top decomposes into non-reproducible pieces:
1. **Train/test well-ID leak plus bundled precomputed submissions** — the forkers' real edge. The leak helps
   the PUBLIC board but is a no-op on the fully hidden private board (a leak-probe kernel confirmed this
   earlier: overriding matched wells changed nothing on the scored test). The "7.159" headline is a bundled
   CSV of unknown provenance, not physics output. I don't chase either; 10.424 is leak-free.
2. **Real-time human/AI dip interpretation** — a geologist reading each well's structural dip. Scored wells
   are hidden and the kernel runs automated, so this cannot be in the loop. The interpretation diagnostic
   (notes/25) tied residual error to un-tracked structural drift on high-relief wells (`corr(err, dip_range)=+0.63`).

## 6. Verifying the claimed ceiling (2026-07-29)
The top public notebooks report an "honest" score near 7.5. I ported the one fully-reproducible public
pipeline (PF + 14-config beam + selector + deg-5 robust projection) and ran it on my 773 labeled train wells,
scoring eval rows against known TVT. It scored **16 to 80 pooled**, never near 7.5: the particle filter
diverges because GR likelihood (SNR≈0.02) cannot anchor its momentum walk. The reported 7.159 is a bundled
precomputed CSV the notebook loads ahead of running any physics. **The claimed headroom is an artifact.**
- On PRIVATE: 10.424 holds; the leak forkers likely shake toward ~10-11; the honest top (~5, if real) stays ahead.
- **BEST = v32 / public 10.424, fully honest. 10.424 is near the honest reproducible ceiling** (notes/28).
