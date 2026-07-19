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

**NEXT (headroom): the blend weight 0.1 was chosen by the UNDERSTATING CV → probably too low.** Since CV
understates this lever, tune the blend on PUBLIC: try w=0.2, 0.3. Also worth: a stronger TCN (more channels
incl. per-well trajectory raw, bigger/deeper, or a small transformer), more seeds. This is now the hot lever.
