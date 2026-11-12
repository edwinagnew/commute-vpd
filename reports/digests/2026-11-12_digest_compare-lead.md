# compare-lead, 2026-11-12

The neuron overlap on Qwen describes the updates but separates neither contrast: every one of the 19 saved updates
keeps fewer of its most active neurons than a random update of the same size, yet RL and its shuffled-reward null read
the same, as do the two maths pools. Only the effective rank separates RL from its null, by a small margin. Every
compare-lead cell is now filled or closed.

## Cells
| Cell | Due | State | Runs |
|---|---|---|---|
| P3.6 cheap readings on Qwen updates | 11-13 | filled | 2/2 |
| P2.6, P2.22 | 11-12, 11-13 | filled yesterday | 1/1, 0/0 |

Ladder: L2.1 and L2.3 answered; L2.2 waits on L1.3; L2.4 waits on L1.5. Nothing is left on a ready rung.

## What ran and what it found
- **P3.6 run 2** (job 59473, cluster debug, $0, 41 min; every control held). The random floor was read from four
  draws, as you chose in chat this morning. RL keeps 0.89 of its top neurons against 0.997 for random; the pools keep
  0.62 against 0.979. RL against null differs by 0.03 seed ranges on overlap and 1.5 on effective rank.
  `reports/analyses/2026-11-12_analysis_p3-6-qwen-cheap-readings.md`.
- The premise check came back doubtful: the rank gap may follow update size, and without the share beside it the
  result cannot say what the share adds. For the 11-18 gate.

## Log, spend, friction
Log v329 published: the P3.6 card is in and the overlap-floor decision card is out. FYI: the flush added 57 words to
the main page, over the "no longer" rule, and well under the 20,000 cap; the rest of that section is other leads' text.
Spend: $0. Friction noted: 743 (the ask 18 card did not say whether run 1 was good or what run 2 was for), 745 (a job
log cannot be committed through the cluster/logs symlink).

## Not run
Effective rank with update size held fixed. The share beside the Qwen readings.
