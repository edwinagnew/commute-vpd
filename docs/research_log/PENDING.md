# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.

- 2026-10-29 baseline-lead · reports/analyses/2026-10-29_analysis_e52-baseline-neurons.md (43d919a, independent check) · e52 card, decision-e52-p4 card · narrow "no small set of neurons carries either" to "the most-shifted neurons up to 256 carry at most 0.24 of the gain"; copy target after 16 examples is about 3e-4 to 4e-4 (1 in 3,000), not 1 in 10,000; "within 0.03 of zero" -> "within 0.031".
