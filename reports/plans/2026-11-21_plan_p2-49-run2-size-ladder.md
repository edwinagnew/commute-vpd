# P2.49 run 2: a size ladder for the crosscoder's planted control at block 2

Cell P2.49 (compare-lead, run 2 of 2, due 2026-11-23); E-number pending (`fleet/ecode`). Appendix (pre-mortem,
settings, cost arithmetic, rehearsal): `reports/plans/2026-11-21_plan_p2-49-run2-size-ladder_appendix.md`. Card:
`~/commute-data/scratch/compare-lead/p249-run2/card.md` (passes, 3.02 SE at n = 3). Rules:
`experiments/crosscoder_proper/results/run2/predictions.json`. Code: `run2.py`, `score2.py` beside run 1's `run.py`.

**The request in the Terms' words** (whip-lead, ~21:15 2026-11-20, from the user). Run 1 (E235) found that at block 2's
input (site A) the authors' crosscoder does not see a grafted head: its fine-tune-specific latents carried 0.30, 0.21
and 0.37 of the head's change over 3 seeds, against the filed 0.5 bar. So block 2 reads "not applicable at this update
size". Run 2 asks at what size it does apply. The head's output weight is multiplied by one factor per rung, so its
activation change per token at A equals the rung's size. The same crosscoder is trained, cheapest rung first, until a
rung passes on every seed. At that rung, X, E147's five 96-key tasks (copy, rank1, elicit, cipher, prior_cipher; 3
saved seeds each) and a Gaussian random change (3 draws) are each scaled to size X and read by the authors' own output.
The result is a stated range: "not applicable below X; at X the five tasks read ...".

## Design check
- **Goal.** Part 2, rung L2.1: the crosscoder baseline's entry. It turns run 1's "not applicable" into a range.
- **Nearest tried.** E235 run 1. Rung 3 is its head at scale 1, reused, not rerun. E237 scaled the same five tasks to
  one common size by weight norm. Here the size is activation change at the site, because a grafted head has no dW.
- **Patch or step.** A patch: run 1's last open question, and the cell's second and last run.
- **Decisions and definition.** No departure. No verdict under the working definition: the pairs are called against
  how they were built, as in run 1.
- **What changes what we do.** If no rung passes, the crosscoder line ends at block 2. If X is near a fine-tune's size,
  the scorecard row gets a reading at X. If X is ~1,000 times larger, the range is the result.
- **Control first.** Stage 0 runs alone: the head at scale 0, which is the base against itself through the new scaled
  path. It fails only on a code fault.

## What runs
| Stage | Items (crosscoders at A) | Reads |
|---|---|---|
| 0, alone | `sizes` (forwards: the head's factor per rung, each task's own size), then `self.A.c0` | L0 |
| 1, round 1 | seed 0 at rungs 1, 2, 4, 5 (2.1, 21, 2,110, 21,099 per token), 4 pods at once | L1 |
| 1, round 2+ | the candidate's seeds 1-2. The candidate is the lowest rung not failed whose seed 0 passed. The other 2 pods start stage 2 at the candidate | L1 |
| 2, at X | 3 random draws first, then 15 task states, 4 pods | G1, G2, tasks, pairs |

The rungs are run 1's head size (210.99 per token) times 10^(i−3). Rung 1 is about a fine-tune's size (elicit 1.96,
cipher 3.22 at A). One seed under the bar fails a rung, so the low rungs need only seed 0. `score2.py ladder` lists the
next keys after each round.

## Expected values and filed predictions
| Number | Self (rung 0) | Rungs 1-2 | Rung 3 (run 1) | Rungs 4-5 | Random at X | Tasks at X |
|---|---|---|---|---|---|---|
| specific part | 0.0, count 0 (stop otherwise) | < 0.3 (high) | 0.30/0.21/0.37 | ≥ 0.5, lean, 0.4 at rung 4 | < 0.5 (medium-low) | ties random on ≥ 3 of 5 |
| clean Pile loss (base 2.71) | 2.71 | ~2.7-3 | 7.45 | ≥ 7.45 | measured | measured |
| relative norm deviation | ≤ 1e-3 | — | — | — | — | — |

- **Range (lean).** Not applicable at block 2 below about 2,100 per token, about 1,000 times a fine-tune's size. X
  passes only where the planted change leaves the model's Pile loss many nats above the base's. X none: 0.30.
- **G2 at X** (the random draws read above every reuse-built seed, run 1's rule): lean fail. Both pairs then read "not
  applicable at rung X" (medium).
- **Blind to.** The specific part does not see a change in how the model uses a direction it already represents. The
  size does not see a change's direction. The range does not see sizes between rungs (a factor of 10).
- **Noise floor.** Run 1's crosscoder-seed spread at rung 3 is 0.21-0.37 (sd 0.081). The rung rule uses a fixed bar,
  and the task rules are ordinal.

## Stop rules
L0 miss: stop, no rung runs. Any stop other than an unreachable size: the ladder reads "stopped", and the lead is told
before the next paid round. No rung passes after round 1: stage 2 does not run, and the pods end. The early checks are
run 1's, plus three run-2 checks before step 0: the scaled path at 0 reproduces W0, the size matches within 5%, and the
change writes at the site.

## Cost and time (appendix C; H100 SXM $4.40/h, ≤ 4 pods)
Measured in run 1: 27.7 min per site-A crosscoder (`results/run1/pod0_{2,3,4}.log`). The size search adds 0.4 min,
from the Gaussian items (34.8-35.3 min against 34.3-35.1). Pod set-up took 25-29 min (start in `fleet/cell show P2.49`
against each pod log's first line); with 0.7 GB more uploaded this is 30 min. The hand-back takes ≤ 5 min.

| Stage | Wall | Pod-minutes | Cost |
|---|---|---|---|
| Rehearsal (cluster `debug` L40) | ~20 min | — | $0 |
| 0: pod 1 set-up, sizes, self; pods 2-4 set up meanwhile | 61 min | 151 | $11.1 |
| 1 round 1 (4 items) | 28 min | 113 | $8.3 |
| 1 round 2 (2 ladder + 2 stage-2 items) | 28 min | 113 | $8.3 |
| 2 rest (16 items, 4 rounds) and the hand-back | 118 min | 471 | $34.5 |
| **All, if the first candidate passes** | **3.9 h** | **848** | **$62** |

- Each failed candidate adds a round: 28 min, $8.3. The worst case, rung 4 failing on seed 1 or 2, is $70 and 4.4 h.
- If no rung passes in round 1: 1.6 h and $20.
- Two crosscoders per card are not used: run 1 measured a 0.94 gain (`results/run1/timing.json`).

## Gates (gated: $62 is over $60)
1. **Pre-mortem.** Appendix B.
2. **Settings sourced.** Every setting has a `_source` in predictions.json. All are run 1's filed settings except the
   rungs, the random seed and the random profile.
3. **Rehearsal (pending, $0).** `cluster/crosscoder_proper_run2_rehearsal.sbatch` on the cluster `debug` L40. At
   20:0x 2026-11-20, `sinfo` showed `compute` up (mix), not down. It covers what run 1's rehearsal (jobs 62445, 62446,
   62452) did not:
   - the scaled head, including scale 0 and the top rung;
   - E147's states and a random draw at the largest factors;
   - e58 elicit's windows;
   - `score2.py`.

   It runs at full read sizes with a tiny dictionary. Then `fleet/rehearsed` records it.
4. **Early checks.** Run 1's, in code, plus the three run-2 checks above.
5. **Review.** One fresh Opus agent, plan and code together: placeholder.
6. **Go-ahead.** Whip-lead, for an in-plan gated run.

## Choices for the lead (they change the reading)
1. **Size unit.** The size is the activation change per token on each side's own read rows (run 1's unit), not E237's
   weight norm or a common text. The size on the head's Pile rows is recorded beside it.
2. **The spliced-loss bar.** Run 1's bar (recovered fraction within 0.05 of self) applies at X as filed. At X it may
   leave the tasks unread.
3. **G2 at X.** It compares against the built reuse sides (rank1 and copy) in place of e54 elicit.
4. **Rungs a factor of 10 apart.** The range has the resolution of one decade.

## Not run
Site B (passed in run 1). Sizes between rungs. Crosscoder seeds 1-2 on the tasks. E147's other three maps. The nine
built pairs and twins at X.
