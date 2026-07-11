# Approach 03 — Level Regression (EDA-driven)

**Status:** concluded-kept (current best, 4.0% over ffill)
**Date:** 2026-07-12
**Config / commit:** `configs/level_regression.yaml`

## 1. Idea & motivation
An EDA-first session (thinking as geologist + data scientist) reframed the problem.
Prior approaches (01, 02) chased GR-wiggle matching — but a variance decomposition
showed that's the *smallest* part of the error. Predict the per-well **level offset**
(eval_mean − last_known) directly instead.

## 2. The EDA that drove it

### Geosteering signature (variance vs lag)
Eval-zone TVT variance grows **super-linearly at short lags then saturates** — not a
random walk (ffill-optimal) nor free trend (extrapolation-optimal), but a **bounded,
mean-reverting drift**. Physically: the driller holds the bit within a target bed.
This explains why every extrapolation attempt (dip, slope, polynomial) diverged, and
why ffill isn't beaten by much locally.

| lag | observed var growth | random-walk baseline |
|----|----|----|
| 100 | 71× | 10× |
| 1000 | 1549× | 100× |
| 2000 | 1779× (saturating) | 200× |

### Error budget (global RMSE, oracle decomposition)
| Predictor | RMSE | meaning |
|----|----|----|
| ffill | 15.910 | baseline |
| oracle constant | 9.035 | perfect LEVEL |
| oracle linear | 6.697 | + perfect trend |
| oracle quadratic | 5.342 | + curvature |

⇒ **LEVEL = 68% of variance, TREND = 15%, WIGGLE = 18%.** We had been optimizing the 18%.

### Signals for the level offset (std 12.69 ft)
- Pre-PS TVT velocity `v50` = (TVT[-1]−TVT[-50])/50 : **r = 0.27** (strongest single signal)
- Projecting velocity forward is optimal at only K≈100 rows (mean-reversion confirmed)
- GR mean-match offset (`gr_off`): weak alone but complementary
- Ridge on {v50, v200, gr_off, eval_len, known_range}: **r=0.32, 10% of offset variance**

## 3. Method
1. `extract_level_features(well)` — v50, v200, gr_off (calibrated mean-GR typewell match),
   eval_len, known_range. All from known zone + typewell (identical train/test, no leakage).
2. `fit_level_model(train_wells)` — Ridge(α=1.0) predicting offset = eval_mean − last_known.
3. `level_regression(well, model)` — pred = last_known + predicted_offset (constant per well).
   Optional `shape_weight` adds a fraction of the Kalman within-zone shape (negligible gain).

## 4. Validation
Grouped 5-fold CV (fit on train-fold wells, predict held-out wells' eval zones — a well
never appears in both train and val):
| Model | Global RMSE | Δ vs ffill |
|----|----|----|
| ffill | 15.910 | — |
| ensemble (Approach 02) | 15.558 | −2.2% |
| **level_regression** | **15.266** | **−4.0%** |
| + kalman shape (w=0.7) | 15.261 | −4.1% (not worth complexity) |
| oracle level ceiling | 9.035 | −43% |

## 5. Contribution
| Component | Effect |
|----|----|
| Ridge offset vs +1.42 global bias | 15.27 vs 15.83 — the per-well signal is the win |
| v50/v200 velocity features | dominant (r=0.27 alone) |
| gr_off feature | small complement to velocity |
| within-zone wiggle shape | ~0.005 RMSE — essentially nil |

## 6. Uncertainty
- Recovers only 10% of offset variance; 68%-of-error level component is far from solved.
- Ridge is linear + regularized → low overfit risk; coefficients stable across folds.
- Velocity signal assumes test wells share the geosteering dynamics of train (reasonable).

## 7. Physical meaningfulness
High. The model says: "the bit was moving at velocity v at the PS point; it keeps drifting
briefly (mean-reversion horizon ≈100 rows) then settles — so the eval-zone mean sits a
predictable offset from the last known depth." This is literal geosteering physics, not
metric-chasing.

## 8. Conclusion
Reframing from wiggle-matching to level-prediction doubled our improvement (2.2%→4.0%).
The remaining headroom (15.27 → 9.04 oracle) is entirely in predicting the level offset
better. Next: richer level features — the .png cross-sections encode bed dip that would
sharpen the velocity signal; also try quantile/robust offset targets and per-well
mean-reversion-horizon estimation.
