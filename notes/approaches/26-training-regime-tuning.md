# 26 — Training-regime tuning: step-5 subsample vs full-rows+strong-reg — 2026-07-19

The one CV-HONEST tuning knob left: our step=5 subsample-as-regularization vs the sub-9 teams' full-rows +
strong-regularization + big-model regime. CV-honest because it's a model-regularization knob, not a blend
(grouped CV tells the truth here, unlike blend weights which it overstates). Same base (rich+formation),
fixed blend w=0.75, single seed for a fair regime comparison (`tools/full_rows_sweep.py`).

## RESULT: our current regime is the BEST. Tuning offers no gain. CLOSED.
| regime | train rows | grouped-CV OOF |
|---|---|---|
| **cur_s5_reg1** (step=5, reg_λ=1, nl=63) | 757k | **10.9422** ← best |
| s5_reg5 (step=5, reg_λ=5, l2=10) | 757k | 10.9693 (+0.027) |
| s2_reg5_big (step=2, nl=127, reg_λ=5) | 1.89M | 10.9639 (+0.022) |
| full_reg_big (step=1, nl=255, reg_λ=5, ne=1200) | 3.78M | 10.9553 (+0.013) |

**More regularization hurts (+0.027). More rows + bigger model hurts (+0.013 to +0.022).** Our step=5
subsample at reg_λ=1 is the best operating point. (Among the alternatives, full-rows+big beats partial —
so IF you use dense rows, go big — but all are still worse than our simple subsample.) Confirms the
long-standing "model is not the bottleneck" and validates the notes/10 subsample-as-regularization choice
with the current rich feature set. The sub-9 teams' full-rows regime is NOT better for our setup.

## Tuning is now comprehensively exhausted (all three categories tested)
1. **Blend/weight/correction-scale**: grouped CV shows −0.10 to −0.20 headroom but it's a MIRAGE —
   transfer-negative 4× (v19: −0.10 CV / +0.03 public) — notes/25.
2. **Model hyperparameters / training regime**: CV-honest, tested here → our regime is already optimal (+).
3. **More seeds** (3→6): safe variance reduction but ~0.01, below the ~0.03 public noise floor.

**Verdict: no tuning headroom. BEST STAYS v20/11.020.** Combined with the 6 feature/method levers closed
this session (notes/20-25), the solution is comprehensively at its honest ceiling. The remaining gap to the
sub-9 cluster is the train/test well-ID leak (non-transferable to private) + real-time human interpretation
(impossible in an automated notebook) — not anything reproducible we have failed to tune or build.
