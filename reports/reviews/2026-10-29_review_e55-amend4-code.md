# Review: E55 Amendment 4 code (2026-10-29)

A fresh Opus agent that wrote none of it reviewed commits c9e2a42 and 65499a3 (`experiments/pcfg_anchors/amend4.py`,
`interchange.pair_index`, `predictions_amend4.json`, the sbatch stage) before the run. Read-only, model-free checks
through cpu-run. All fixes went in at d62c43f, before job 44746 started, with a dated entry in the predictions'
`amendments`.

Checked and fine: `pair_index` makes the same random draws as before, so `pair_up` gives identical pairs for run.py and
amend1-3. Off-label rows are built right (both swap directions, |swap| against the text before the swap, filters on
the counts after it). Pairs hold the other count. Seeds do not collide. The natural reproduce check uses a2's seeds and
prefix. The sbatch stage is sound.

| # | finding | severity | done |
|---|---|---|---|
| 1 | Part 2 rule needed count b "not through" (P_b < 0.100); W1 1K reads b "neither" on its own natural rows (P_b 0.104), so a pure count-a answerer would read "neither" | should fix, near blocking | rule changed: "reads a" iff a through and b not through; mirror for b |
| 2 | P1s (seed copy agrees) passed on void = void or neither = neither | should fix | void unless W1 1K reads a or b |
| 3 | LOW in Part 2 reads "neither" by construction; not a control. No count-a positive control on swapped text | nit | LOW made a description; the missing control stated |
| 4 | Bases clip answers to 10..17; 6.7% of off-label rows had count a + 1 at 18-19 | nit | rows kept only with count a + 1 in 10..17 |
| 5 | Part 1 cannot separate the readings | nit | already described only |
| 6 | amend4.json written only at the end | nit | written after controls and after each part |

Earlier, the code writer had found that the lead's design for Part 1 (whole-residual patch at block 5, natural donors,
swap-built recipients) reads "a" under both readings, because the patch there copies the donor's answer. The lead made
Part 2 (off-label swap-built rows) the scored part before filing (65499a3).
