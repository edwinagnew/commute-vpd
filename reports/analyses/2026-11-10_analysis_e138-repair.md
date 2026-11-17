# The repair run that completed the first test of the second reuse wording

*2026-11-10, cell P1.6. The readings are on the page of cell P1.3, `reports/analyses/2026-11-10_analysis_reuse-variable-first-test.md`.*

**In plain words.** The first test of the second wording stopped twice before it finished, once on a solver the card's library refused and once on a lack of card memory. The repair run fixed both without changing the design or the filed predictions, and ran the test to its end at no cost on the cluster. Its reading is the first test's, for example a found re-weighting share of 0.922 on the Always on pair.

## Fixes, each listed

- The rank of the original model's activations moved to a last step, on the MAGMA solver, after cuSOLVER refused both solvers (jobs 57027 and 57034; commits `4bc054d7` and `429c7040`). A failure there would lose only that unscored number; it ran.
- The ridge matrix is freed before the solve, and the time is checked before each H site, after the card ran out of memory at the `final` site (job 57043; commit `d29b519a`; job 57100 solved `final`).

The repair run is jobs 57142 and 57208, the second a resume after the first reached its time limit.

## Not run

No other repair. The H refinement failure noted on the P1.3 page was not fixed: a fix would be a new run.
