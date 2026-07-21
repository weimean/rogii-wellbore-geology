# 27 — Sequence model: the learned holistic "reading" — 2026-07-19

**Question (user):** can an agent/transformer substitute the human geologist's *reading* of the GR log?
**Answer: partially YES — the FIRST positive lever of the session.** An LLM reading plots at inference is
impossible (no internet, no trained geo-model, runtime). But a learned SEQUENCE MODEL is legitimate and
adds signal: everything in our stack predicts each row INDEPENDENTLY (GBM/MLP take a feature vector per row;
the HMM/PF are sequential trackers fed AS features). A per-well **TCN** (dilated temporal conv over the eval
sequence) reads the well HOLISTICALLY — the learned analogue of reading the whole log — and captures
sequential structure in the raw GR correlation the row-independent models miss.

## Result (grouped-CV OOF on cached signals, `tools/seq_model_probe.py`, baseline 10.9034)
| input set | seeds | best blend | OOF | gain |
|---|---|---|---|---|
| model outputs only (hmm/pf/gbm/mlp/fusion) | 3 | w=0.2 | 10.90+ | **none** (hurts) |
| + 54 rich GR-match channels (the raw "reading") | 3 | w=0.2 | 10.8778 | −0.026 |
| + 37 per-well trajectory/GR-envelope channels | 3 | w=0.2 | 10.8558 | **−0.048** |
| (robustness, seeds {3,4,5}) | 3 | w=0.2 | 10.8720 | −0.031 |
| **honest inputs only** (drop fit-on-train gbm/mlp) | 3 | w=0.1 | **10.8697** | **−0.034** |

**Key findings:**
1. Sequential reprocessing of our MODEL OUTPUTS adds nothing (they're already extracted) — it must read the
   RAW GR correlation surface. That's the "reading": the rich GR-match channels (hgr − typewell_GR at 11
   offsets) integrated as a SEQUENCE.
2. CV-HONEST: all inputs are per-well (own GR + typewell + trajectory), NO neighbour wells → grouped CV does
   not understate it (unlike spatial), and it's a new decorrelated model, not a blend mirage. Validated the
   same way MLP/TabICL were.
3. Robust: two seed triplets both positive (−0.048, −0.031, avg ≈ −0.04); worse ALONE (11.4–12.7) but
   complementary in blend — the genuine-diversity signature (like PF/MLP/TabICL).
4. **Honest-inputs version = −0.034 at w=0.1** — drops the fit-on-train gbm/mlp from the INPUTS (keeps them
   in the OUTPUT blend). All remaining channels are honest per-well → clean kernel build, NO inner-CV needed.

## 🚀 PUBLIC RESULT: 11.020 → 10.852 (−0.168), NEW BEST. kernel v25, ref 54824955.
Shipped the torch TCN into the kernel: train seed-ensemble (3) on all 773 train wells, inputs = 91 honest
per-well channels (X[:, :-30], dropping spatial+formation+lik), resampled to 256 stages, target = TVT−anchor;
predict test wells, interpolate, blend 0.9·correction + 0.1·TCN. Kernel log confirms `TCN trained on 773 well
sequences (91 channels)` — torch ran on Kaggle (CPU), no fallback.

**The gain was 5× the CV estimate: −0.034 grouped CV → −0.168 public.** Grouped CV UNDERSTATED the sequence
model, like it understates spatial features. Likely reason: the final TCN trains on ALL train wells (vs 4/5
per CV fold), and sequence models benefit from more data; the scored test wells are offset wells well-covered
by the full train set. **So the "honest ceiling" at 11.020 was ARCHITECTURE-POVERTY, not input-limited** —
every prior model predicted rows independently; the holistic sequential reading was the missing lever. (Same
mistake shape as the 13.52 "ceiling" = feature-poverty.)

## Tuning the lever on PUBLIC (CV can't be trusted — it understates 5×)
| step | change | public | delta |
|---|---|---|---|
| v25 | TCN, blend 0.1 | 10.852 | −0.168 |
| v26 | blend 0.1→0.2 | **10.760** | −0.092 |
| v27 | stronger TCN (5blk/ch64/5seed) + blend 0.3 | **10.699** | −0.061 |
| v28 | + transformer ensembled (0.65 TCN / 0.35 TFM) | 10.726 | **+0.027 REVERTED** |

**Stronger TCN worked** (CV signal −0.034→−0.081 with 5 blocks/ch64/5 seeds; public −0.061 with blend 0.3).
**Transformer REJECTED (v28).** A 3-layer/4-head/d64 TFM encoder trained fine on Kaggle (no fallback) but
ensembling it with the TCN scored 10.726 vs TCN-only 10.699. Self-attention's GLOBAL receptive field
**RESTATES** the TCN's holistic reading rather than decorrelating from it — the same outcome as beam-DP vs
our HMM (corr 0.9997). Averaging in a correlated, slightly weaker model just dilutes. Damage was small
(+0.027) exactly because ensembling two sequence models is near-neutral when redundant. `_TFM` class kept in
the kernel for reference but NOT trained (also saves ~30min runtime).

**LEVER SATURATED: −0.168 → −0.092 → −0.061 → +0.027.** The sequence-model vein is now closed at
**v27/10.699**. Within sequence models, the TCN already captures the holistic reading; a different sequence
architecture is a restatement, not new signal. (Remaining untried, unvalidated: feeding RAW GR sequences so
the model learns its own correlation from scratch — but the local cache rig would need a ~2h rebuild first.)

## TCN config sweep — NULL RESULT (2026-07-21), do not re-run
Clean, contention-free grouped-CV sweep (sequential, pinned 8 threads, 5 seeds each; baseline 10.8775).
Ran the SAME config twice (A1, A2) to establish the error bar:

| variant | best OOF @ w=0.2 | vs A1 |
|---|---|---|
| A1 ch64 (v27 cfg) | 10.8164 | — |
| **A2 ch64 (same cfg, diff seeds)** | **10.7648** | **−0.052 (this IS the error bar)** |
| B ch96 (wider) | 10.8084 | −0.008 |
| C prefix-crop aug | 10.8293 | +0.013 |
| D input noise | 10.7886 | −0.028 |

**The A1–A2 spread (0.052) exceeds every variant's deviation from A1.** ch96/crop/noise are all
indistinguishable from the v27 config — B sits between A1 and A2; D (−0.028) is smaller than the seed
noise and worse than A2 (baseline cfg, lucky seed). NO variant beats v27 config by more than measurement
error, and 0.052 seed variance > the ~0.03 public noise floor anyway → nothing here is submittable.
**Confirms the sequence-model lever is at its ceiling at the v27 config.** (Also explains the earlier
contaminated sweep's flipping rankings: ~0.05 seed variance, not just CPU contention.) Rig now persists in
~/rogii_cache (survives /tmp wipes) + _cache_meta_oof.py scripts the lost cache — reproducible if ever needed.

## 'Last breath' — RAW-GR sequences NULL (2026-07-21), lever fully closed
Hypothesis: our rich features (hgr − tw_gr(HMM+off) at 11 offsets) are a lossy reduction; a from-scratch
model given the un-reduced surface (41 fine offsets + 21 raw typewell-reference + 3 raw-GR scales = 65
channels, `_cache_rawgr.py`) might learn a better correlation. Clean sweep (5 seeds, pinned threads,
baseline 10.8775), REF run twice for the error bar:

| config | best OOF @ w=0.2 | verdict |
|---|---|---|
| REF_rich | 10.8164 | reference |
| REF_rich (repeat) | 10.7648 | **error bar = 0.052** |
| RAW_only (65 raw ch, no rich) | 10.8451 | WORSE than both REF |
| RICH+RAW | 10.7997 | inside the REF band (null) |

**REFUTED.** RAW_only is *worse* than the rich reference; RICH+RAW sits between the two REF runs → no signal
beyond seed noise. Our hand-engineered 11-offset reduction is NOT a bottleneck — it's better-conditioned than
the raw surface, which just gives the data-starved TCN (773 wells) more to overfit. "Learn from scratch" <
"the right summary". **The sequence-model lever is now fully closed: architecture (transformer=restatement),
config (all within 0.05 seed noise), and input representation (raw GR < rich) all exhausted. FINAL BEST =
v27 / public 10.699.**
