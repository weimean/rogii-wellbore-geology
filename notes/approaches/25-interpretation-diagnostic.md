# 25 — Human/AI interpretation: diagnose where we fail, encode the insight — 2026-07-19

**Constraint:** the scored wells are HIDDEN and the notebook runs AUTOMATED at scoring time (Kaggle re-runs
it on the private test set, no human present). So literal human interpretation of the scored wells is
impossible — any interpretation must be encoded into the automated algorithm. Legitimate version: act as the
interpreter on TRAIN wells, diagnose the geological failure mode, encode a generalizing correction.

## What the interpretation found (I looked at the 6 worst wells — `tools/err_interpret_diag.py`)
Pooled OOF 10.90; per-well error median 6.9, p90 15.4, max 52. Error is concentrated, and the strongest
predictor is **dip_range (structural relief over the eval zone): corr(error, dip_range) = +0.63** (worst-80
wells median relief 43 ft vs best-80 median 18 ft). GR_std and eval_len barely matter (+0.13, +0.15).

**The failure mode (visible in `worst_wells.png`):** on high-relief wells the true TVT DRIFTS across the
long eval zone (a slow continuous structural dip, +60/−40 ft), but our prediction reverts to the FLAT anchor
and misses the drift. The GR is varying on these wells (there IS signal), but our tracker follows the drift
too conservatively.

## Why we're flat, and why it's (mostly) correct
The drift DIRECTION is unpredictable: pooled signed residual ≈ +0.17 (near 0) — the drift is symmetric
(some wells up, some down), while |residual| grows heel→toe (2.9→10.3). `corr(|resid|,|true_drift|)=+0.69`:
our error IS the un-tracked drift. Because direction is unpredictable, the GBM correctly regresses drift
toward 0 → flat. The high-relief wells are hard because their drift is large AND unpredictable.

## The one encodable lever — de-shrink the correction — and why I did NOT ship it
`anchor + k·(pred−anchor)` improves grouped CV: k=1.15→10.82, k=1.2→10.81 (−0.09). BUT:
- This is the **exact blend-family knob that grouped CV OVERSTATES and public has REVERTED 4×** (adaptive
  blend; fusion-weight cut v19 = −0.10 CV / **+0.03 public**; TabICL blend; prefix-CV). For correction-scaling
  knobs, grouped CV is optimistic (opposite of spatial features, which it understates).
- Decomposing: amplifying the per-well TREND is razor-fragile (x1.2 helps −0.04, x1.4 HURTS +0.19, x1.6
  catastrophic) — because the drift direction is unpredictable, amplifying magnifies wrong-direction wells.
  Shrinking the wiggle hurts (it's real tracker signal). So the −0.09 is mostly wiggle amplification, fragile.
- On the real test (offset wells, MORE novel/noisy), the correction SHOULD be MORE shrunk, not less. The
  grouped-CV gain is the same mirage that fooled us 4×.

## Conclusion
The interpretation SUCCEEDED diagnostically: the residual error is un-trackable structural drift on
high-relief wells. But encoding a fix hits the known wall — predicting that drift needs signal we don't have
(direction unpredictable; the tracker/spatial levers are at parity/sparse), and the one knob that flatters
grouped CV (de-shrink) is a fragile, transfer-negative blend change. **The genuine value a human geologist
adds — confidently reading each well's dip in real time — is exactly what an automated notebook cannot do,
which is why the top teams' edge is real-time interpretation (impossible here) or the well-ID leak.** BEST
STAYS v20/11.020. De-shrink NOT shipped (prior: −0.09 CV → likely + on public, like v19).
