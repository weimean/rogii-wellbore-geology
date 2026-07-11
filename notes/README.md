# Working notes

Day-to-day thinking lives **here in the repo** (versioned next to the code that produced
it). Durable, cross-competition insights get distilled **up** into the Obsidian vault note
at milestones — this folder is the source, the vault is the curated summary.

| File | Purpose |
|------|---------|
| `eda-findings.md` | What we've learned about the data & wells (facts, verified numbers, domain mechanics) |
| `approaches/_TEMPLATE.md` | Copy per new approach — structured on the Working-Note rubric |
| `approaches/NN-name.md` | One deeply-analyzed approach each (idea → validation → conclusion) |

## The rubric (why the template looks like it does)

Structure every approach the way the competition's Working Note Award asks (the award
deadline passed 2026-07-06, but it's the best rubric going):

1. **Idea & motivation** — what and why.
2. **Validation result** — grouped-by-well CV RMSE, honestly reported.
3. **Conclusion** — why it worked or failed. Negative results count fully.
4. **Contribution** — quantify each component's effect on CV.
5. **Uncertainty** — where the method is reliable vs. shaky.
6. **Physical meaningfulness** — real geology vs. metric-chasing.

Prefer a few deeply-analyzed approaches over many shallow ones. Hyperparameter tweaks and
window-size changes are *the same approach*, not a new one.

## Distill-up checklist (repo → vault, at milestones)

- New durable insight? → add to the vault note's **Key Findings**.
- New reusable domain concept? → spin off a `02 Concepts/` note (e.g. `[[Well Log Correlation]]`).
- Approach concluded? → one-line verdict in the vault **Experiment Log** table (detail stays here).
