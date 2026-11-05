# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
- 2026-11-05 · reports/analyses/2026-11-05_analysis_e90-old-labels.md, notebooks/E90_new_vs_reuse.ipynb, experiments/TRIED.md · experiments (E90 card), timeline, overview's definition line, decisions-row context · E90's table of the old 67M labels under the frozen definition, provisional, no independent check: only E62's pair reads as assumed through a located mechanism; under direction removal cipher and random map read "reuses" with no C listed and rank-1 reads "not specific" by 0.02; rows labelled existing stay unread; filed R2 lean reversed; a graded order at block 1 is post hoc
