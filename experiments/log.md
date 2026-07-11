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
