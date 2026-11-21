# compare-lead digest, 2026-11-21 (overnight)

P2.52, due 11-23: filled. Neither baseline sorts the five synthetic key tasks the way the log-odds share does: the
neuron shares set copy apart, and E203's crosscoder sets no task apart and reads most fine-tunes below a random change
of the same size. P2.51 is the only open cell, held until you decide on wording A.

## Cells (`fleet/cell list compare-lead`)
- Filled overnight: P2.49 (E243), P2.50 (E237), P2.52 (E244), P2.53 (E246); E235 written in full.
- Open: P2.51 (cheap proxies against R under wording A, due 11-23, no runs). It is on hold: definition-lead's P1.46
  (E242) failed its known-answer test under wording A, so there is no R under A to score against yet.

## What ran and what it found
- **E244 (P2.52).** The neuron baseline and E203's crosscoder, controls first, on E237's five tasks with three seeds
  each. Every control passed. The whole-width neuron share is lowest on copy and the top-neuron share is highest there,
  but the top 64 neurons carry under three hundredths of the gain on every task. The crosscoder reading varies by up to
  four tenths between seeds of one task, and 13 of 15 fine-tunes read below their matched random band. The plan's
  expected copy reading of about 0.06 rested on E203's single seed; the other two seeds read about 0.49 (a miss,
  stated in the report). The premise check was doubtful on the crosscoder column (it may not be reading these updates
  at all); listed for the gate. Report `reports/analyses/2026-11-21_analysis_e244-key-grid-baselines.md`, figure
  `experiments/key_grid/figures/columns.png`.
- Research log republished as v428 with the E244 card.
- Saved crosscoders of E243 and E244 moved into `results/large`; leftovers in `~/commute-data/stale/compare-lead/2026-11-21/`.

## Waiting on you
- Wording A (via whip-lead): unblocks P2.51.
- Sign-flip proxy proposal (note in whip-lead's folder): no cell yet.
- Idea only, not run, no cell: a second crosscoder seed on copy seed 0 and top seed 2, to tell the fine-tune's
  effect from that one crosscoder's training.

## Money
No pod this session ($0.00); the earlier session of this lead spent $152.46. No machines on record.

## Friction noted
974 (`fleet/numbers` misreads a jq key like `.E203`), 975 (`commute.figures` has no crosscoder style); open from
11-20: 970 (pod label rule vs briefs), 971 (job-gate refuses any command containing "ladder").

## Not run
The second-seed crosscoder check above; anything on P2.51.
