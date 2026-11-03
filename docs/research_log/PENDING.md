# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
- 2026-11-03 · reports/analyses/2026-11-03_analysis_e78-anchor1.md, notebooks/E78_qwen_anchor1.ipynb · experiments card E78, timeline · ran: anchor 1 retrained at E61's rate reproduces E61 (step 100 within 1.7e-6, all 256 answers byte-identical, template 92.6%, strict accuracy 62.1%, `learned` true), with fp32 δW at steps 1, 2, 3, 5, 10, 20, 50, 100 ready for the split readings; step t has t − 1 moving steps, so later labels are one moving step behind E69's.
