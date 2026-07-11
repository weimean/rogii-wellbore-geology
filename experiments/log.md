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
