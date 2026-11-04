# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
- 2026-11-04 · reports/analyses/2026-11-04_analysis_e79-masked-read.md, notebooks/E79_qwen_masked_read.ipynb · new E79 card; pending-decision card "After E76, which change do we carry on with?" changes (its recommendation waited on E79, and the running line goes); claims table; "What's next" D1 paragraph; timeline · A low reading under recomputed masks is no evidence of placement: do-nothing reads 0.066 against the clean-term change's 0.078, and that change's masks on the unchanged weights read 0.094 against 1.334; it can penalise harm, it cannot credit placement; the all-masked and routed changes read mixed under frozen masks and do not survive on web text; the clean-term change has shown only clean agreement at 0.52 of the update's squared size; the clean table cannot fail do-nothing, for the user. New figures in experiments/qwen_masked_read/figures/.
