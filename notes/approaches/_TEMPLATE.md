# Approach NN — <short name>

**Status:** idea | in-progress | concluded-kept | concluded-dropped
**Date:** YYYY-MM-DD
**Config / commit:** `configs/<file>.yaml` @ `<git sha>`

## 1. Idea & motivation
What is the approach, and why should it work here (tie to a data/domain finding)?

## 2. Method
Enough detail to reproduce: features, reference used (pre-PS / typewell / offset), alignment
technique, any post-processing.

## 3. Validation
Grouped-by-well CV RMSE. Report per-fold + mean. Compare against the current best and the
baseline floor.

| Variant | CV RMSE (mean ± sd) | LB RMSE | Notes |
|---------|---------------------|---------|-------|

## 4. Contribution
Quantify each component's effect (ablation): what does adding X buy in CV RMSE?

## 5. Uncertainty
Where is this reliable vs. shaky? (e.g. long gaps, short pre-PS anchors, sparse GR).

## 6. Physical meaningfulness
Is this recovering real geology, or fitting the metric? Where's the line?

## 7. Conclusion
Keep / drop, and **why**. What did we learn (even if it failed)? What does it suggest next?
