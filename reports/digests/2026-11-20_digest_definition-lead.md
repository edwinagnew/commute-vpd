# Definition-lead digest, 2026-11-20

P1.36, due 11-21: filled. The reuse score on the trained pairs mostly follows update size, and with no level checked against a known answer it is not yet the paper's label column. One ask waits on you: the next cell (VPD featurization).

## Cells
Part 1 has no open cell. 29 filled; closed: P1.5, P1.10, P1.11, P1.21, P1.24, P1.31 (`fleet/cell list definition-lead`). P1.35 (built items, E214) and P1.36 (trained pairs, E221) were filled on 2026-11-19.

## What ran and what it found
- **E221 (P1.36), 8 RunPod H100 pods, about $27.8.** The reuse score was read on 80 saved fine-tunes of fourteen matched pairs, at three coarse levels: heads and MLP layers, MLP and attention layers, and blocks. Each pair's reuse-minus-new gap ranks with its update-size gap (Spearman 0.54, 0.69 and 0.45) more than with any other reading, but under the filed 0.7 bar, so P1 missed. No earlier reading agrees with it on every pair, and the held-model reading ranks against it (P2 held). The smallest circuit that carries the behaviour is nearly the whole model at these levels (all six blocks on every fine-tune), which leaves the score little room. Report: `reports/analyses/2026-11-20_analysis_e221-kstar-pairs.md`.
- **Premise check: doubtful** (`~/commute-data/premise/P1.36.md`). In E214 (P1.35) the hand-built reuse edits could not be built, so no level has been checked against a known answer. Its question for you: what result would show that the score measures reuse rather than update size?
- **Owed FYI:** E208 run 2 cost about $80-90 in all, over the $60 gated line. Searches took 12-15 min, against the 6-21 planned, and one runner's pods idled through a slow W&B fetch.

## Asks waiting on you
- **55.** A new cell, due 11-22, as your 2026-11-19 task asked: read the reuse score at the VPD featurization (subcomponents plus each site's remainder) on E187's nine pairs. The options:
  1. Recommended. In one job: the do-nothing control, then the six known-answer items (E214's two built new edits and E195's four reuse items), then the nine pairs. Two runs.
  2. The nine pairs only, one run.
  3. Option 1 at E217's pruned-neuron featurization instead of VPD.

  Card: `reports/decisions/2026-11-19_decision_vpd-r-cell.md` (log v412).

## Money
This session: none since E221. E221 was about $27.8. No pods are live and no Slurm jobs are running.

## Friction noted
913 (log section-number drift), 919 (E205's silent file-presence branch), 922 (sync sends every untracked folder to pods; job-gate pod-label form), 923 (writers cannot save drafts to scratch).
