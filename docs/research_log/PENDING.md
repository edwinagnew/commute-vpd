# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.

- 2026-11-02 baseline-lead · experiments/TRIED.md (E59 row), experiments/jain_readout/results/predictions.json · E59 card (planned → coded) · E59 coded and its predictions filed: a per-block readout trained on Pile text (never on the task tables), read at the elicit and cipher keys, and removal of its direction in the twelve saved E54/E56 fine-tunes (E56 did not save the cipher's own-step states for seeds 2-4, so that check uses seeds 0-1); one cluster debug job, $0, not yet run.
- 2026-11-02 · reports/plans/2026-11-02_advice_c2-filing-FABLE.md, predictions_c2.json · E55 card · C2 predictions filed; noise floor 0.020 means the resume check passes and cannot see an error of that size; ~56 L40-h
