# Experiment Log (terse, code-side)

Append one line per run while working. The richer narrative + decisions live in the
Obsidian vault note `04 Projects/Kaggle — ROGII Wellbore Geology Prediction.md`.

Metric: **MSE** (lower is better). Always report **grouped CV** (by WELLNAME).

| Date | Exp id | Change | CV MSE | LB MSE | Keep? | Notes |
|------|--------|--------|--------|--------|-------|-------|
| 2026-07-11 | baseline_ffill | carry last known TVT forward | — | — | ? | pipeline smoke test |
| 2026-07-11 | baseline_linear_interp | interpolate TVT over gap by MD | — | — | ? | honest floor to beat |
