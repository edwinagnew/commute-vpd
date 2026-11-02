# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.

- 2026-11-02 qwen-decomp-lead · reports/reviews/2026-11-02_review_qwen-pipeline-audit.md · E57 card, E45 card, Mistakes · pipeline audit: the direct share's "original masks" read the moved read vectors (the prose was wrong; size on the share not yet measured); size has two axes that disagree 2.5-fold; switched-on sets are subcomponents just under the gating line, so "three to four times chance" falls to about 1.0-1.3 (settled from files); re-gating near 1 has an algebraic floor, no leak, "no teeth" stands; own run's +0.052 is less separated than reported. Five $0 checks follow. (v161 added a one-sentence mention to the E57 card only; the E45 card, Mistakes and the overlap figures remain.)
