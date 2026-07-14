# ROGII Wellbore Geology — Final Conclusion

**Task:** predict eval-zone TVT (True Vertical Thickness) for horizontal wells. Metric: RMSE.
**Final model:** `gbm_stack` (`configs/gbm_stack.yaml`) — seed-ensembled LGB+Cat — **CV 13.499 / public 13.549 (best on both)**.
**Verdict:** at the input-limited ceiling. The remaining error is structurally irreducible with
the data available at inference.

---

## 1. The model (what we ship)

Anchor-then-residual, diverse-algorithm stack (all inputs test-available):

```
per well ── LEVEL ANCHOR (Ridge on v50,v200,gr_off,eval_len,known_range) ─┐
        ├── HMM posterior  (2nd-order (TVT,dip-rate) grid, fwd-backward)   ├─ variance FUSION
        └── PARTICLE FILTER (bootstrap, different algorithm)               ┘   (trust HMM where
                                    │                                            hmm_std low)
              _stack_features (~34 cols: HMM/PF posteriors, trajectory, GR,
              anchor/fusion, uncertainty group A, GR-envelope group B)
                                    │
              GBM STACK: mean(LightGBM×3 seeds, CatBoost×3 seeds) on target = TVT-anchor,
              trained on every 5th eval row (subsample = regularization)
                                    │
              final = 0.6·(anchor + mean(6 boosters)) + 0.4·fusion
```

Progression (773-well GroupKFold CV): ffill 15.91 → level_regression 15.33 → hmm_fusion 14.22 →
gbm_stack(HistGBM) 13.77 → +PF 13.53 → +A/B 13.54 → LGB+Cat @ step=5 13.52 → **+seed-ensemble = 13.50**.
Seed-ensembling removes the single-seed lockbox lottery (spread ~0.08) and reduces variance.

## 2. Why 13.5 is the ceiling (all proven by measurement, notes 09-11)

Error budget: **68% LEVEL + 15% trend + 18% wiggle**. Oracle bounds: constant 9.04, linear 6.70.

| Component | Status | Evidence |
|----|----|----|
| LEVEL (per-well offset) | geology-locked | 4 levers (velocity, spatial, GR-align ×2) all die at offset R²≈0.10; full stack reaches 0.337 by confidence-weighting |
| TREND (per-well slope) | geology-locked | OOF R²≈0 from velocity/Z/GR; pre-PS drift doesn't continue (driller steers back) |
| WIGGLE | saturated | HMM+PF+LGB+Cat; residual has NO directional signal in any available feature |

**The deepest reason (`tools/tops_probe.py`):** TVT = f(Z, formation tops) with per-well R²=1.0 —
geology *perfectly* determines TVT. But f is per-well (each well has its own geological datum), and
the tops are **train-only**. Even ORACLE eval tops add only +0.014 to the offset without the per-well
frame. The eval level is a per-well constant defined in a reference frame the test set withholds.
Residual analysis (`tools/residual_diag.py`): error grows 7.3→17.1 with eval depth; median well RMSE
is 8.6 (≈oracle) — the pooled 13.5 is a tail of long, undirectable laterals.

## 3. The leak, resolved (notes: leaderboard-intel memory)

- The 3 wells in `data/raw/test/` are **format examples** that happen to be identical to train wells
  (leaked). They are **NOT the scored set** — the real test is hidden.
- **The public LB is a trustworthy hidden test that tracks CV** (phase-2 CV 13.52 / public 13.60).
  Proven by submitting a twin-matcher leak probe (`kernel_leakprobe/`): it overrides any test well
  that matches a train twin with the twin's true TVT — and scored **13.604 = phase-2's 13.601**, i.e.
  the real scored test has **no exploitable duplicates**. Trust both CV and public; they agree.
- **Geology was a leakage trap** (notes/11): typewell `Geology` is train-only; it gave −0.53 CV but
  the feature is absent at inference → public got worse (13.93). Reverted.
- **Competitor 7.016 notebook** (evaluated in full): its anchor family = ours; its gains rely on a
  same-well contact override (yields nothing on the real test, per our probe) and its novel ideas
  (datum projection, spatial-tops) fail on our CV (imputed tops → 51 RMSE). Likely pretrained-artifact
  based; not reproducible honestly.

## 4. Reproduce / operate

- CV table: `uv run python -u tools/reproduce_cv.py` (~15 min).
- Fast experiment harness: `tools/frontier_fast.py` (cached matrix + train subsampling).
- Caches (wiped on reboot): `_cache_base.py` → `/tmp/rogii_feats.npz` + `rogii_pf.npz`.
- Submit: kernel `gtwm98/rogii-hmm-fusion` (v8 = current seed-ensemble); the tools index is `tools/README.md`.

## 5. Honest final position

We are near the true structural limit for these inputs. There is no dishonest-free headroom of
consequence: level+trend (83%) is geology the test withholds, wiggle is saturated, and the LB is
trustworthy so there is no leak to mine. The deliverable is a rigorous, well-validated model **and**
a measured proof of its ceiling.
