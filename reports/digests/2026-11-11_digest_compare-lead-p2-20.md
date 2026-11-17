# compare-lead, 2026-11-11 (evening): P2.20, due 11-12, filled

**Five of five part 1 pairs overlap the method table, on the same saved fine-tunes.** Part 1's re-weighting share puts
the side built to reuse higher on all five. The re-gating share agrees on the three built pairs and makes no call on the
two label twins. Steering ranks the five pairs exactly as part 1 does but reads both twins the other way. No method
agrees with part 1 in sign on all five.

| Method | agree / reverse / no order / missing | rank correlation, n |
|---|---|---|
| re-gating share | 3 / 0 / 2 / 0 | 0.7, 5 |
| prior | 1 / 0 / 4 / 0 | 0.7, 5 |
| steering, strength 1 | 3 / 2 / 0 / 0 | 1.0, 5 |
| SVD of the update (effective rank) | 2 / 2 / 1 / 0 | 0.8, 5 |
| update size | 0 / 4 / 1 / 0 | 0.5, 5 |
| neuron, whole width | 1 / 3 / 1 / 0 | -0.1, 5 |
| neuron, half | 1 / 2 / 2 / 0 | 0.3, 5 |
| neuron, top neurons | 0 / 0 / 5 / 0 | 0.1, 5 |
| share on the untrained decomposition | 1 / 2 / 2 / 0 | 0.2, 5 |

Page: `reports/analyses/2026-11-11_analysis_p2-20-part1-vs-methods.md`; figure
`experiments/part1_vs_methods/figures/part1_vs_methods.png`; rule filed first in
`experiments/part1_vs_methods/results/predictions.json`. Saved results only, no job, $0.

- The method-table counts come from `final.json`'s tally. The P2.11 report's steering count of 5 predates E147's refit;
  it is 4 (dated note, commit 32715f45).
- Neuron readings on the two label twins filled later the same evening by E158 (cell P2.21, job 58971, controls pass, $0): case label no call; repeat label the shuffled twin reads higher on every seed at both widths, the reverse of part 1. No reading is now missing on the five pairs.
- Premise check "doubtful" (advisory, for the 11-18 gate): on direction, part 1 repeats the build labels, so only the
  ranking of gap sizes is new, and on five pairs (three for the re-gating share) that is thin.
- Descriptive only: no reused/new verdict while the definition is under revision (P1.1).

Ladder: L2.2's two cells (P2.20, P2.21) are filled. P2.21 premise check also "doubtful": on a real-versus-shuffled twin, "reuse side" means learned versus noise, so agreement there says a method detects learning, not reuse (for the gate). Open: P2.6 (parked by the user). Asks waiting on you: none. Spend $0.
