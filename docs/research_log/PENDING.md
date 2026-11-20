# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
A line `CONSENT YYYY-MM-DD <role> · <the user's words>` records that the user waived the main-page word cap for this flush (README.md → Main-page length; friction 688).

- 2026-11-19 · reports/analyses/2026-11-19_analysis_e217-topk-neurons.md · cells table, part 2 neuron baselines (section 4.2.3) · P2.42 filled (E217): with the base model's weak MLP activations pruned (p97/p3 per-neuron thresholds) as the featurized model, the definition reads 0.96-0.99 on all 20 trained fine-tunes, below the ~1.00 at full width; two filed leans reversed: fine-tunes change which neurons fire strongly far more than expected (overlap 0.23-0.36, filed 0.7-0.99), and the pooled weak firing changes more than expected (remainder 0.83-0.94, filed >=0.99). Descriptive, no verdict; premise check doubtful (neurons may not suit this model's updates).
