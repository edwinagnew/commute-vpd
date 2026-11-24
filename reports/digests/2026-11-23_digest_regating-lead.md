# regating-lead digest, 2026-11-23

P2.66, due 11-24: the cell is filled, and I am finishing because I have no open cell and no rung. The day's last run,
E279 (the untrained-decomposition share on the realistic panel), gave no call. Whether re-gating comes back into the
paper waits on the user's claim-1 decision. Whip-lead records it as "option A" for now, with a final call tomorrow
morning.

## Cells
| Cell | State | Runs used / allowed | Due |
|---|---|---|---|
| P2.33 | filled | 2/2 | 11-19 |
| P2.34 | closed | 2/2 | 11-19 |
| P2.36 | filled | 0/0 | 11-21 |
| P2.40 | filled | 0/0 | 11-21 |
| P2.66 | filled | 3/6 | 11-24 |

## What ran today
- E276 was the confirmatory text panel. Effective rank, scored after taking out base loss, came out at a partial
  correlation of 0.89. But 7 of the 12 tasks were kept, and the tier ordering and base loss were collinear (0.82),
  so the headline is "not separable". redteam-lead judged it.
- The E272 E_b scores came home overnight ($84.60). Over the needed set, E_b ranks the realistic tasks in the
  reverse of the built direction. redteam says the reversal holds, and that base loss explains it within text.
- E279 is the last P2.66 refill. The untrained-SVD share does not order the panel (tau 0.03), so it gives no call.
  Page: `reports/analyses/2026-11-23_analysis_e279-untrained-panel.md`.

## Asks waiting on the user
None of mine. The E_b decision (A, B or C) is whip-lead's ask.

## Spend
This session (since the 13:35 restart): $0, with no pods and no jobs. The overnight E272 E_b pods were billed to
earlier sessions; whip-lead keeps the tally.

## Friction noted today
1022 (sbatch --wrap runs under sh), 1025 (the job-gate label form), 1028 (rsync is missing on fresh pods),
1029 (execute() calls to parallel pods run one after another, and RunPod deleted 2 of 4 pods), 1030 (the log flush's
python3 route is refused).
