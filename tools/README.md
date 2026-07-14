# tools/ index

## Production / reusable
- **reproduce_cv.py** — canonical 773-well GroupKFold CV table (the headline numbers). ~15 min.
- **frontier_fast.py** — fast experiment harness: cached feature matrix (`/tmp/rogii_stackX.npz`),
  vectorized build, per-well train subsampling. Import `load_stack`, `train_submask`, `oof`.
- **frontier_cv.py** — the LOCKBOX harness (dev 80% / lockbox 20%, seed 42). Gates features on held-out wells.

## Investigation (one-off probes — kept as the record of the ceiling proof)
Level / trend:
- **level_probe.py** — offset recoverability; spatial K-NN (dead).
- **level_headroom.py** — how much offset variance the full stack recovers (0.337).
- **level_align.py** — global + band-limited typewell GR alignment for level (dies, GR aliasing).
- **slope_probe.py** — per-well trend/slope recoverability (R²≈0, geology-locked).
- **tops_probe.py** — TVT = f(Z, formation tops), per-well R²=1.0; but per-well frame (oracle +0.014).

Geology (all concluded — feature is train-only):
- **geology_probe.py** — typewell Geology → level (oracle 0.187; looked like a win).
- **geology_stack.py** — geology in the full stack (CV −0.49 — but it was leakage).
- **critique_diag.py** — bootstrap CI + ablation + test-well representativeness (caught the leak).
- **geology_infer_probe.py** — infer Geology from shared cols (marginal, 0.113).

Frontier features / model:
- **frontier_features.py** — Group A (uncertainty), B (GR envelope), C (NCC, inert).
- **phase2_fast.py / phase2_metastack.py / phase2_lockbox.py / phase2_final.py** — LGB+Cat @ step=5 selection.
- **ensemble_robust.py** — seed / 3-way ensemble robustness for the private LB.
- **shrink_test.py** — uncertainty-aware shrinkage (no gain; fusion already optimal).
- **residual_diag.py** — where the residual lives; geosteering interaction (residual-R²≈0).
- **tune_hmm.py** — HMM coordinate-descent tuning (overfits).

Competitor / leaderboard:
- **datum_test.py** — the 7.016 notebook's U=T+Z projection (catastrophic on our data).
- **spatial_tops.py** — 7.016's spatial-tops idea: oracle 0.007, imputed 51 (unrecoverable).
- **leak_test.py** — proved public LB ≠ the 3 leaked example wells.

## Cache builders (root)
- **_cache_base.py** → `/tmp/rogii_feats.npz` + `rogii_pf.npz` (HMM/PF posteriors, ~8 min).
- **_cache_frontier.py** → `/tmp/rogii_frontier.npz` (GR envelope group B).
- **_cache_geology.py** → `/tmp/rogii_geology.npz` (geology feats; reverted feature).

See `notes/CONCLUSION.md` for the full write-up.
