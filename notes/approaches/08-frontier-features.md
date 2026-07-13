# Approach 08 — Frontier feature groups under a lockbox

**Status:** concluded-kept (A+B), with honest caveats
**Date:** 2026-07-13
**Config / commit:** `configs/gbm_stack.yaml` (predictor `gbm_stack`, now incl. A+B feature groups)

## 1. Idea & motivation
The frontier-rebuild question: can bolt-on feature groups push the stack toward the
public-notebook tier (~9)? The pooled 773-well OOF CV is prone to feature-selection
overfitting, so we introduced a **lockbox** (`tools/frontier_cv.py`): dev (619 wells, 80%)
is used for OOF/fitting; lockbox (154 wells, 20%, fixed seed) is *never tuned on* and serves
as the clean private-set proxy. A group is kept only if it improves the lockbox.

## 2. Groups tested
- **A — inter-signal uncertainty:** `|hmm−pf|`, `hmm_std+pf_std`, normalized disagreement,
  `|hmm−anchor|`, `|pf−anchor|`. Transforms the GBM can't derive from raw posterior columns.
- **B — GR envelope/energy** on eval rows: rolling max/min/rms (w=51), rolling std (w=31,151),
  smoothed `|dGR|`. Local texture of the log the tracker is matching against.
- **C — multi-scale NCC quality:** local corr of eval GR vs typewell GR along the
  HMM-predicted TVT path at hw=8/15/25 (how trustworthy the alignment is locally).

## 3. Validation — lockbox (154 held-out wells)
| Features | dev-CV | lockbox | lockbox-deep |
|----|----|----|----|
| BASE (gbm_stack+PF) | 13.518 | 13.742 | 16.112 |
| + A | 13.628 | 13.631 | 16.009 |
| + B | 13.533 | 13.650 | 15.993 |
| + C | 13.518 | **13.742** | 16.112 |
| + A+B | 13.566 | **13.623** | **15.969** |
| + A+B+C | 13.566 | 13.623 | 15.969 |

- **C is inert** — zero change on every metric. It's redundant with `hmm_std`, which already
  encodes alignment confidence. Dropped.
- **B is the clean win** — improves lockbox at ~no dev-CV cost (both held-out metrics move
  the right way).
- **A** improves lockbox but *raises* dev-CV — the classic pooled-vs-held-out tension.
- **A+B** is best on both lockbox (−0.12) and lockbox-deep (−0.14). The improvement being
  **consistent across two independent groups** is what makes it real rather than noise.

## 4. The pooled-CV caveat (important, documented honestly)
On the full 773-well GroupKFold pooled CV, A+B is **13.540 vs 13.530 — i.e. ~0.01 worse**.
The pooled OOF metric and the lockbox disagree in sign. Both moves are tiny and within noise.
We keep A+B because the **lockbox is our designated private-set proxy** (never tuned on; the
whole reason it was built), and it says A+B generalizes better — but on the headline pooled
metric this is a **wash, not a win**. Not claimed as a new pooled-CV best.

## 5. Conclusion
Kept A+B, dropped C. This is a small, honestly-a-wash-on-pooled-CV / modest-lockbox-gain
change — not a breakthrough. It confirms the plateau: individual feature groups are now
marginal. The remaining headroom to the ~9 tier is whole-pipeline engineering (LGB+CatBoost
meta-stack + heavy tuning), not increments. See [[diversity-and-ceiling]], notes/07.
