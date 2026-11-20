# qwen-lead digest, 2026-11-20

P3.13 (due 11-25) is filled, 3 of 3 runs. The held-model reading on Qwen's pair library now uses a fit that
reproduces the model on every row set, and no pair separates at the bar. That is rival B as filed. No cell of mine
is open.

## Cells
- P3.13: filled, 3 of 3 runs (the user raised it from 2 today). Page:
  `reports/analyses/2026-11-20_analysis_p3-13-held-readings.md`.
- Every other part-3 cell is filled or closed. Rung L3.2's only cell is filled. Only you change the ladder, and it
  still lists L3.2 as open.

## What ran
- E225, on RunPod: Slurm was down, and you said speed comes first.
- **Rehearsal.** Fitting on 64 own rows plus 64 maths rows brought the held model's do-nothing error on the word rows
  from 3e-4 to 2e-3 down to 2e-9 or less.
- **Scored pods.** Two pods then reread copy against cipher and known against invented.
- Run 2's floors, the stage 0 sweep and the per-seed gaps are in `experiments/qwen_held_readings/results/run3/summary.json`;
  the figure is `figures/run3_S.png`.

## What it found
- **Late save, both sides at saved size.** The side built to use an existing skill reads higher: about 0.059 for
  copy over cipher, 0.012 to 0.020 for known over invented.
- **Late save, size-matched.** With the larger side shrunk to the smaller side's size, the copy/cipher gaps fall under
  0.005. The known/invented gaps straddle it.
- **Early (5x) save.** The gaps are under 0.001.
- **Random references.** On word rows they read 0.94 to 0.97, against about 0.81 on maths rows.
- **A filed expectation missed.** We expected shrinking a change to lower S, as on the 67M. Here it raised S: cipher
  went from 0.93 to 0.99. The cause has not been checked.
- **Premise check: doubtful.** On Qwen, S puts every fine-tune at 0.98 to 0.997 and separates no pair. So the reading
  may not tell conditions apart on this model, and section 6 may be reporting a ceiling.
  - The checker asks: before these readings, what result on Qwen would have counted as part 2 failing to repeat?
  - The full check is in `~/commute-data/premise/P3.13.md`.

## Money
E225 cost $6.61 on pods: rehearsal $1.34, copy/cipher $2.89, facts $2.37. That is out of the session's $300 budget.
No pod is running.

## Asks
None open.

## Friction noted
- `fleet/rehearsed` takes its arguments in a strict order and wants the words "per item" in its text.
- `fleet/numbers status` cannot price the first pod of a run, the one whose label has no stage word.
