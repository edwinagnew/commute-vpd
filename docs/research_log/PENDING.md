# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving
this list empty. Recipe: `README.md` → "Batched updates".
- 2026-10-27 · reports/analyses/2026-10-27_analysis_broad-fragility.md, notebooks/E39_broad_fragility.ipynb · e39 card (new: E39 has no card yet; figures moves/reverts/shape), decisions (decision-broad-fragile was decided and removed in v60, so only its outcome goes in the e39 card and next-commute; decision-pass-anchor: the random-move control ran and passed), timeline · E39: the trained decomposition's failure is not weight-size brittleness (random moves of matched size pass, 1.03/1.07, though they do ~1/25 of the damage) and not the rarely important subcomponents (their half removes 0.63 of the excess vs 0.97 for a random half); the MLP changes carry it together (reverting MLP 1.00, attention 2.70), and a rank-8 LoRA's trained decomposition passes (1.27; stable rank 1.9 vs 28.5, but half the damage); SGD learned at no grid rate, no state, Amendment 1 with smaller rates planned
