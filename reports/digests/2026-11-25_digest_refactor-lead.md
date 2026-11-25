# refactor-lead digest, 2026-11-25

**Cells:** P2.79 (reading refactor, due 11-26) is filled, using 1 of its 3 runs (E309, $0 on the cluster debug queue). No other cell is open, so I am finishing.

**What ran.** The reading refactor is merged to main (08f49906), with whip-lead's approval for the switch-over. Every reading the paper uses now has one definition in `src/commute/`. 50 finished experiment folders moved, unedited, to `archive/experiments/`. Old and new code agree on the same saved states: in 50 of the 51 table rows they match exactly. The remaining row, one per-key steering value, lies inside the old code's own run-to-run spread; whip-lead read it as a pass under the cell's rule. Every paper figure file that can be rebuilt came out byte for byte the same. One fresh Opus review was applied; its main finding was that archived scripts still loaded by live code would have broken, so those folders stay live. Report: `reports/analyses/2026-11-25_analysis_p2-79-reading-refactor.md`.

**Limits.** Existing run scripts were not edited, so old scripts still load each other; new code reads from `src/commute/`. Gaps between saved results and recomputed ones that the old code shows too were sent to writeup-lead, along with two older faults on main: a stale figure builder and two paper CSVs that differ from every repo copy.

**Asks waiting on the user:** none. **Money:** $0; no pods. **Blocked:** none.

**Friction noted:** 1154 (three sbatch stage tests already fail on main; a test collection error on main, fixed in the refactor); 1155 (moving folders breaks the research log's figure sizes until the README figure list is updated).
