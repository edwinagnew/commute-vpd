# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
- 2026-11-05 · decisions.md · decisions, e84, timeline · The user chose option 1 of `decision-anchor2-after-e84` (ask 8): the 5-epoch update stands as the Qwen example of content the model lacks, labelled "new facts memorised, carrying only partly to rephrasings". Remove the card; note the choice on E84's card; the comparison's second column can now be read.
- 2026-11-05 baseline-lead: decision report reports/decisions/2026-11-05_decision_new-vs-reuse-definition.md: Jain's new-vs-reuse definition was not settled; user decided 'new' is relative to a named capability and direction removal is the causal label; admissible capabilities and constant targets open (ask 19).
