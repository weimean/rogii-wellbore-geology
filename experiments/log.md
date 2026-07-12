# Experiment Log (terse, code-side)

One line per *run* (quantitative ledger). The per-*approach* narrative
(idea→validation→conclusion) lives in `notes/approaches/NN-*.md`; milestone verdicts get
distilled up into the vault note `04 Projects/Kaggle — ROGII Wellbore Geology Prediction.md`.

Metric: **RMSE** (lower is better; Kaggle scores √MSE). Always report **grouped CV** (by WELLNAME).

| Date | Exp id | Change | CV MSE | LB MSE | Keep? | Notes |
|------|--------|--------|--------|--------|-------|-------|
| 2026-07-11 | baseline_ffill | carry last known TVT forward | — | — | ? | pipeline smoke test |
| 2026-07-11 | baseline_linear_interp | interpolate TVT over gap by MD | — | — | ? | honest floor to beat |
| 2026-07-11 | baseline_linear_interp | interpolate TVT over gap by MD | — | — | floor | eval zone = well tail → degenerates to constant last-known TVT; pipeline verified vs sample_submission |
| 2026-07-11 | kalman_gr_v1 | EKF tracking TVT via typewell GR + 1.42ft bias | 15.735 | — | ✓ | 1.1% over ffill (15.910); GR→TVT signal weak (SNR~0.03), beats ffill on 51% of wells |
| 2026-07-11 | gr_offset_v1 | calibrated mean-GR typewell lookup, α=0.16 σ=100 | 15.581 | — | ✓ | 2.1% over ffill; best standalone model |
| 2026-07-11 | ensemble_v1 | 0.25×kalman + 0.75×offset | 15.565 | — | ✓ | 2.2% over ffill; best overall |
| 2026-07-12 | level_regression_v1 | Ridge on per-well level offset (v50,v200,gr_off,eval_len,known_range) | 15.266 | — | ✓ | **best**; 4.0% over ffill, grouped 5-fold CV. EDA: level=68% of error budget; wiggle ~unpredictable |
| 2026-07-12 | ncc_lightgbm_v1 | HistGBM per-row on NCC+trajectory+azimuth+tortuosity feats | 15.247 | — | ~ | ties level_regression; top feat=z_from_ps. NCC standalone=21.7 (quasi-periodic aliasing). Public LB (4.86) = 3 leaked wells, not representative |
| 2026-07-12 | hmm_fusion_v1 | 2nd-order (TVT,dip-rate) fwd-bwd HMM ⊕ level_regression by posterior var (blend_k=4) | 14.239 | 15.116 | ✓ | 10.5% over ffill. LB rank ~3626/4734 (median 9.65, top 4.86) → behind the field. Submitted via kernel gtwm98/rogii-hmm-fusion v2 |
| 2026-07-12 | selfcorr_hmm | add calibrated self-correlation emission ref to HMM | 15.37 | — | ✗ | REGRESSION — calibration distorts emission scale; reverted. A/B was misleading (compared vs calibrated baseline not production) |
| 2026-07-12 | gbm_stack_v1 | HistGBM on [HMM post + level anchor + trajectory] correcting anchor, 0.5 blend w/ fusion | 13.769 | 13.727 | ✓ | **best**; 13.4% over ffill. Public 13.727 rank ~3353/4750 (was 3634); CV≈public now. LightGBM ties HistGBM → features are bottleneck |
| 2026-07-12 | hmm_resolution | sweep s_target/step (250/500/800) | — | — | ✗ | finer HMM HURTS (250=13.82, 500=14.08, 800=14.72); more steps → more drift/aliasing. 250-bin default is good |
| 2026-07-12 | self_ref_hmm | add self-reference HMM (own pre-PS GR profile) + disagreement to GBM stack | 13.768 | — | ✗ | +0.001 (nil). Redundant with typewell HMM already in stack — both match eval GR to a TVT reference. Multi-scale NCC self-corr lever is captured by our HMM |
| 2026-07-12 | hmm_tuning | coordinate-descent tune (mom,sig_p,rate_span,lam,band_pad,sig_r), tune/holdout split | — | — | ✗ | OVERFITS. Tune 13.82→13.52 but HOLDOUT tuned 14.81 vs defaults 14.62 (+0.184 worse). EDA-set defaults near-optimal; tools/tune_hmm.py |
| 2026-07-12 | gbm_stack+PF | add bootstrap particle-filter (TVT,dip-rate) posterior as GBM-stack features, 0.6 blend | 13.530 | — | ✓ | **best**; 15% over ffill. PF worse alone (raw 19.4, fusion 15.2) but a DIFFERENT tracker → complementary as features (13.77→13.53). Validated vs cache + full reproduce_cv |
