# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
- 2026-11-24 · reports/analyses/2026-11-24_analysis_e299-known-3seeds.md · experiments card E299/E303 (known_3seeds), timeline 11-24, main page section on the share against baselines · P2.77 filled (baseline-lead): Fig 4 and known_appendix re-read on 3 seeds a side with one variant a side, the size-matched series on the same states (E299, E303, cluster, $0). The share separates 9 of 9 built pairs; steering reverses the share on 2 of 9 pooled and 3 of 9 per key, and on 3 of 9 when sizes are matched (the third from the size scaling alone). The paper's old count of three mixed the two steering forms.
