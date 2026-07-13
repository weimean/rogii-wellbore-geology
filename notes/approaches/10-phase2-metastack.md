# Approach 10 — Phase 2: subsample regularization + LightGBM/CatBoost diversity

**Status:** concluded-kept (current best; biggest lockbox gain in the project)
**Date:** 2026-07-13
**Config / commit:** `configs/gbm_stack.yaml` (predictor `gbm_stack`, now LGB+Cat @ step=5)

## 1. Idea
After the LEVEL lever closed (notes/09), the user chose to build the wiggle meta-stack anyway
for diversity + LB optics. Two levers turned out to matter — one unplanned:
1. **Training optimization → regularization.** Iterating was slow (full 3.78M rows), so we added
   per-well row subsampling. Training on every 5th eval row didn't just speed things up 4x — it
   *improved* CV: the dense, autocorrelated eval rows were mildly overfitting the booster.
2. **Model diversity.** Replace the single HistGBM correction with an average of two genuinely
   different modern boosters (LightGBM leaf-wise + CatBoost ordered/symmetric).

## 2. Validation — LOCKBOX (154 held-out wells) is the decider
| Blend (step=5 train subsample) | Lockbox | Pooled CV |
|----|----|----|
| current: HistGBM, full data | 13.7699 | 13.5808 |
| HistGBM, step=5 | 13.6719 | 13.5100 |
| LightGBM | 13.5446 | 13.4976 |
| CatBoost | 13.5396 | — |
| **LightGBM + CatBoost** | **13.5050** | 13.4892 |
| Hist+LGB+Cat | 13.5396 | 13.4737 |

- **Subsampling alone**: lockbox 13.770 → 13.672 (−0.10). Non-monotonic sweet spot at step=5
  (step=1 13.581, step=5 13.510, step=10 13.551 pooled) — clean regularization.
- **LGB+Cat**: lockbox 13.770 → **13.505 (−0.265)** — the biggest single held-out gain we have.
- **Drop HistGBM**: adding it *hurts* the held-out blend (Hist+LGB+Cat 13.540 > LGB+Cat 13.505).
  Pooled CV mildly prefers the 3-way (13.474 vs 13.489), but the lockbox — never tuned on — is
  our private proxy and prefers LGB+Cat, so we ship that (also simpler/faster).
- **Learned Ridge-meta** over the base preds did *worse* than a plain mean — don't over-engineer.

## 3. Production
`fit_gbm_stack` trains LightGBM + CatBoost on step=5 rows; `gbm_stack` averages them, then the
usual 0.6·(anchor+pred) + 0.4·fusion blend. Both boosters are on the Kaggle image (internet-off
safe); step=5 keeps the fit fast. `tools/frontier_fast.py` (cached matrix + vectorized build +
train subsampling) is the fast harness that made this iteration tractable.

## 4. Honest reconciliation — the gain is modest, not the −0.265 the lockbox showed
The canonical 773-well GroupKFold CV (`tools/reproduce_cv.py`, LGB+Cat @ step=5) is **13.520**,
vs 13.540 for the old HistGBM+A+B — a **−0.02** improvement, the same ballpark as A+B.

The lockbox's −0.265 is **selection-biased**: we scored 7 blends on the lockbox and picked the
winner (max-of-7 → optimistic), and its baseline (13.77) was a single hard fold with the simpler
anchor. The **canonical GroupKFold CV is the unbiased estimate** (5-fold, every well held out
once, not selected on) — so **−0.02 is the honest expected gain**. Both metrics agree on
direction; the magnitude is modest.

## 5. Conclusion
The two levers the user pushed on — optimize the training (subsample regularization) and
diversify the algorithm (LGB+Cat) — both help and stack, but modestly: **canonical CV
13.540 → 13.520**. This does NOT contradict the level-ceiling finding (notes/09): it's
wiggle/variance-side improvement (the 32% of budget that isn't level), so it won't reach ~9.
Kept as the best config (principled, directionally positive on both metrics), not overclaimed.
See [[diversity-and-ceiling]] (diversity = different algorithm), [[level-ceiling]].
