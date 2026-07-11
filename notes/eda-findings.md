# EDA & domain findings

Granular record of what we know about the data and the geology. Verified numbers only —
guesses are marked. Distill the keepers up into the vault note's *Key Findings*.

## The task in physical terms
Predict **TVT** (geological position, ft) along each horizontal well *after* the
**Prediction Start (PS)** point. `TVT_input` = TVT known up to PS, NaN after. Signal =
**GR-signature matching**: as the bit crosses dipping rock beds, its Gamma-Ray curve
traces a shifted copy of a reference GR-vs-TVT profile. Recovering TVT = finding, per foot,
which TVT makes the horizontal GR line up with a reference.

## Verified data facts (773 train wells, 3 example test wells)
- **Schema** — `horizontal_well.csv`: `MD, X, Y, Z, ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA,
  TVT, GR, TVT_input` · `typewell.csv`: `TVT, GR, Geology` · `sample_submission.csv`: `id, tvt`.
  Formation-top columns (ANCC…BUDA) and `TVT` are **train-only**.
- **Submission = eval-zone rows only.** `id = {WELL}_{positional_row_index}`, rows where
  `TVT_input` is NaN. Sample = 14,151 rows across 3 wells. Verified our builder's id set
  matches sample exactly.
- **PS-is-tail holds for 100% of 773 wells** — contiguous NaN tail, zero interior gaps,
  zero wells lacking a pre-PS anchor.
  - pre-PS length (known head): min **851** / median **1703** / max **2392** rows
  - gap length (to predict): min **407** / median **4840** / max **10052** rows
  - pre-PS fraction of well: min 0.12 / **median 0.26** / max 0.80
  - ⇒ the gap is typically **~3× the anchor**; the drill enters TVT ranges unseen pre-PS,
    so the typewell (full geological column) and offset wells are **required** for the deep
    part of the gap, not optional.
- **Metric = RMSE** (Evaluation section formula; the "MSE" tag is imprecise). `n` = number
  of predicted rows. Report grouped-by-well CV RMSE locally.

## Domain mechanics (task pptx, 14 slides)
- **Directional constraint (slides 6–7):** GR pattern rising along the reference ⇒ TVT
  increasing; falling ⇒ decreasing; **flat GR ⇒ constant TVT** (well running along a bed).
- **⭐ Self-correlation beats typewell (slide 9):** horizontal-well GR has better, more
  consistent resolution than the typewell; the post-PS GR often correlates better with the
  *same well's pre-PS GR* than with the typewell. Strongest reference = own pre-PS section.
- **Offset-well dip prior (slides 10–13):** dips behave similarly in neighbouring wells;
  drilling azimuth affects expected dip. X/Y neighbours give a prior on TVT drift direction.

## Open data questions
- Distribution of GR NaNs within wells (pptx says "some GR may be NaN") — how much, where?
- How well-separated are wells spatially (for offset-well matching)? Need the X/Y map.
- Typewell coverage: does each typewell's TVT range span the full horizontal gap?
- [guess] eval zone likely mirrors train (tail) in the hidden test — confirm on the 3 examples.
