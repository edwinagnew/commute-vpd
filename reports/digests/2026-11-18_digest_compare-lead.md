# compare-lead digest, 2026-11-18 (evening)

compare-lead finished: every cell of mine is filled or closed (33 filled, 3 closed, none open), and the next item, the
updated part 2 gate page, waits on a zero-run L2.3 cell that only the user can add.

## Cells
- P2.38, due 11-22: **filled**, 3 of 3 runs. Real-text pairs on the 67M (topic label vs a seeded coin label, French vs
  Yoruba Wikipedia, lowercase vs reversed word order). The share separated only lowercase vs reversed; it gave no call
  on the label pair and could not read French. No cheaper reading did better. The crosscoder (E212) failed its
  do-nothing control: a random update the size of the French fine-tune read above the stop bar, so the real sides were
  not run; its planted control passed. Report `reports/analyses/2026-11-18_analysis_realistic-tasks-run3.md`.
- Premise check on P2.38 came back doubtful (`~/commute-data/premise/P2.38.md`): with so few calls on real text, the
  paragraph on what the share adds over the cheap readings still rests on the built pairs. For the gate.
- No open cells.

## Waiting on the user
- Add the zero-run L2.3 cell, due 11-20: bring the part 2 gate page
  (`reports/analyses/2026-11-16_analysis_part2-gate-page-update.md`) up to date with everything since 11-16 (size-matched
  pairs, neuron baselines, the crosscoder and its failed do-nothing control, disagreeing pairs, built-in-kind fine-tunes,
  realistic tasks), saved results only, no verdict. You said yes in the dialog (ask 25); my `fleet/cell add --rung` was
  refused by the auto-mode permission check, so it needs you to run it or allow the command.

## Spend
E212: $8.03 on two H100 pods, both terminated. No pods or Slurm jobs live.

## Log
Research log v404 published (P2.38 row filled, E211 card updated with runs 2 and 3 and the premise doubt). Queue empty.

## Friction noted
884 `make test` broken at collection by `src/commute/library_states.py`; 885 an unmeasured 1e-5 construction bar cost a
rehearsal; 887 `fleet/cell add` refused by the auto-mode classifier; 890 the numbers ledger has no source kind for pod
cost.
