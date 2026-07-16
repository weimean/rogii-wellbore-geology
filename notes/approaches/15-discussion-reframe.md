# 15 — Discussion deep-dive: the REAL lever is per-well GR matching (2026-07-16)

Read the ROGII discussion forum + new notebooks. **Major reframe — overturns two of our conclusions.**

## The metric + oracle ladder (confirmed, matches ours exactly)
Metric = **pooled row-RMSE** over all eval rows. Full-773 oracle ladder (Georgy Mamarin's harness,
independently reproduced by wharekawa to 2 dp): **carry-last 15.91 · best-constant 9.04 · best-line 6.70 ·
smooth 3.05**. (Mean-per-well RMSE runs ~30% lower than pooled — watch which people quote.)

## OVERTURNED #1: "GR matching is dead" — WRONG
- **Tucker Arrants (7th): a SINGLE model reaches pooled CV ~5 ft (maybe <4), using ONLY each well + its
  typewell — no spatial, no offset wells, no leak.** k256 (12th): tabular CV in the 5.x, LB consistent.
  shu01 leads at 4.859. These are BELOW the line-oracle (6.70) → they capture the WIGGLE, legally.
- Our memory said "GR SNR≈0.02, matching caps ~1% over ffill." That was matching done WRONG.

## OVERTURNED #2: "18% wiggle saturated" — WRONG. The wiggle is where the top teams win (smooth oracle 3.05).

## The mechanism (how they pin the datum + wiggle legally)
1. **Affine GR gain/offset heel-calibration BEFORE the datum scan** (Georgy §5 v7): raw GR-datum scan
   localizes only **8%**; fit `G̃=(G−β)/α` on the known heel first → **~80%** (oracle 82%). "The 8% was the
   calibration, not the data — the datum IS legally pinnable." Our HMM/PF match RAW GR without this. ⟵ #1 gap.
2. **Bimodal hedge** for GR self-similarity: predict the posterior mean of the two GR-misfit minima instead
   of committing → ambiguous-well datum 5.8→5.1. This is THE central hard problem (APIyuya's CNN/attention
   blew up on exactly these "confidence jumps"). ⟵ #2 gap.
3. **Azimuth / drilling-direction split** (De DQ): NW vs SE wells traverse the typewell GR in REVERSE order;
   one model gets confused. Split/flag by direction — "often drops the score significantly, easiest win."
   (Our old azimuth probe tested it as a DIP predictor — wrong use.)
4. **Huber-clipped datum scan cost** `J(Δ)=mean clip((G̃−G_tw(T+Δ))/s,−6,6)²`; multi-scale; GR-rotation
   denoise (Tabish) nudges 80→84%.

## Spatial is LEGAL (not a leak) — our 11.042 gains are honest
Georgy: the public board is NOT leaky (example wells replaced by ~200 hidden). Within-field spatial
interpolation is a LEGAL signal (his field-held-out split discards it by design). So our −1.11/−0.74 spatial
gains are legitimate. The ~7.3 public notebooks' well-ID override IS a separate leak — k256 predicts they
shake DOWN to ~10 on private. Our path (spatial + GR, no override) is the robust one.
- **Stronger spatial (notebook "Another Approach", honest CV 9.32→9.21): anisotropic variogram KRIGING of
  the per-well datum residual b_w directly** (fitted rotation + 2× axis ratio, K=25, shrink 0.30), not the
  surface. Cleaner than our isotropic nndist gating.
- **Visible-prefix self-calibration**: per-well, label-free backtest on hidden suffixes of the KNOWN prefix
  to select/blend candidate trajectories with bounded moves. We have zero per-well adaptivity.

## Error structure (steven's diagnostics, ~8.8 pooled)
Error splits by nearest same-field neighbour distance: <150 ft → already at line-oracle 6.70; no neighbour
<600 ft → 10.15 and dominates. So: close-neighbour wells = copy neighbour's full TVT curve; isolated wells
= per-well GR matching is the only lever (and it works to ~5 with proper calibration).

## PATH FORWARD (honest, robust, big headroom from 11.042)
1. **Fix GR matching**: affine heel-calibration (G̃=(G−β)/α) + Huber-clipped multi-offset datum scan +
   bimodal hedge. Biggest lever — single models reach ~5 with this.
2. **Azimuth/direction split** — flag updip/downdip; cheap, "easiest win".
3. **Anisotropic variogram kriging of b_w** — stronger honest spatial datum (CV 9.32→9.21).
4. **Visible-prefix per-well self-calibration** — label-free candidate selection.
5. Keep trusting a field-grouped hold-out + public LB; the ~7.3 leak notebooks are fragile (shake to ~10).

## Reproduction attempts (2026-07-16) — the top honest levers DON'T reproduce naively
Tested the three "honest" levers surfaced above. All three FAIL on our data:
1. **GR datum scan (affine heel-cal + bimodal), flat-U shape:** pooled 87 (vs ffill 15.9). The GR cost
   landscape's minimum is NOT at the true datum — exactly steven's forum finding. Needs a good tracker
   SHAPE first (Georgy's 80% is "centred on the true shape"), not flat-U.
2. **Affine-GR offset refinement on the PF shape:** 17.8 → 20.4 (WORSE). Scanning a GR offset on the PF
   position moves it AWAY from truth — the GR signal is misleading at the ±25 ft scale.
3. **Anisotropic kriging of the per-well level L_w:** LOWO L_w-error 13.4–14.6 vs null 12.77 (WORSE than
   predicting zero). Raw per-well level is not spatially predictable at K=25 (regresses to field mean);
   survey-density-limited, per wharekawa/Georgy. The notebook's 9.32→9.21 kriges the MODEL RESIDUAL
   (tiny post-hoc shrink), which our formation-plane's close-neighbour gating already captures.

**Conclusion:** the top teams' ~5-6 ft honest edge is a proprietary GR-matching formulation (Tucker 7th,
shu01 1st won't disclose; APIyuya's CNN/attention blew up on the bimodal ambiguity). The OBVIOUS
reproductions all fail. Our 11.042 (formation-plane + trackers + MLP, all honest) is a solid, ROBUST result
— the ~7 leak notebooks are predicted to shake to ~10 on private (k256, 12th). Remaining honest headroom
needs either the undisclosed GR method or the meta-stack-of-stacks architecture (needs OOF infra rebuild).

## Option 2 (GR method) — dedicated agent, rigorous NEGATIVE (2026-07-16)
Attacked the ~5ft GR lever with windowed NCC, DTW, sequence scans, HMM inversion, affine heel-cal,
bimodal hedge. Full-773 pooled results:
- best GR-match (affine level-pin + bimodal softmax): **14.83 (−1.08 over ffill 15.91)**.
- GR-inversion ORACLE (pick match nearest truth = cheat): **12.2** → NO point/sequence matcher beats ~12.
- horizontal GR vs typewell GR at TRUE TVT: corr **0.65** (35% unexplained) → too self-similar to invert.
- wiggle inversion (local-linearisation dTVT): HURTS (14.6→17.0), it's noise. HMM GR-emission: 26.1.
- Error is LEVEL-dominated (level 12.77, wiggle std 7.5); GR pins the level only weakly (→11.54).
- U=TVT+Z flat-datum scan: 104 (wrong physics — in the toe TVT~flat while Z wanders 100-200ft).
**Verdict:** GR matching is a small lever (−1) already subsumed by our multi-offset GR-match family. The
forum's ~5 is NOT a cleverer typewell-GR formulation (physics caps ~12); it's spatial/offset-well (we have
it) or prefix-CV leakage. Probes: tools/gr_window_probe.py, gr_level_probe.py, gr_seq_probe.py.

## Option 3 (meta-stack) — MARGINAL (2026-07-16)
Positive-Ridge meta over [p_gbm, p_mlp, fusion] fit out-of-fold, + MD-ramp fade. Pooled 773-well CV:
- current fixed blend: 10.9034
- positive-Ridge meta: **10.8728 (−0.031)**, weights [0.826 gbm, 0.112 mlp, 0.062 fusion]
- + MD-ramp fade τ=60: 10.8721 (fade adds ~nothing)
Real but tiny (near public noise floor). The signal: the meta wants MUCH less fusion (0.06 vs our 0.25) and
less MLP — echoes the known "fusion over-weighted because the GBM ingests HMM/PF as features." A fusion-
weight reduction (0.25→~0.10) may capture most of it more simply. tools/meta_stack_test.py.

## BOTH TRACKS EXHAUSTED — 11.042 stands
Option 2 (GR) rigorous negative; option 3 (meta) marginal −0.03. No meaningful reproducible honest lever
remains beyond 11.042. The exploration's lasting value: (1) our 11.042 is HONEST + ROBUST (leak notebooks
predicted to shake to ~10 private); (2) the top ~5-6 is proprietary GR + within-field spatial we can't fully
crack; (3) our blend slightly over-weights fusion. Recommendation: bank 11.042 (kernel v18).
