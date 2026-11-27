# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
- 2026-11-26 · reports/analyses/2026-11-26_analysis_e314-sae-proxies.md (cell P2.85, E314 and E317) · experiments; the P2.85 row flushed in v492 · Correction and addition to the P2.85 line: the states are the known-answer figure's (known_main; the cell calls it Fig 3), 16 tasks x 3 seeds, not "Fig 4's". E317 (the control): with only the fine-tune's SAE reconstruction patched in, the base restores 0.40-0.98 of the gain; the error and reconstruction shares sum to 1.26-1.64 on every task, so the two parts are not additive and the error share above one half does not show the gain lies outside the dictionary. Neither share, nor Kissane et al.'s gap, separates tasks built to reuse from tasks built new; appendix G and the tools table.
