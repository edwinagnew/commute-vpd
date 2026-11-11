# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
- 2026-11-11 · reports/analyses/2026-11-11_analysis_p2-16-gap-sizes.md · E133 card (baselines), next.html P2.16 row, timeline · P2.16 filled: across the nine built pairs the share's gap sizes follow neither the prior's gaps (rank correlation 0.02) nor strength-1 steering's (-0.10), n=9, descriptive; figure experiments/share_gap_sizes/figures/gap_sizes.png. Next: P2.17 (2 runs, RunPod up to $15, user 2026-11-11) builds pairs where prior and steering disagree.
- 2026-11-11 · reports/decisions/2026-11-11_decision_built-control-sum-stop.md · built-control card (E148's), pending decisions · P1.10 closed: the sum-parity built control (E150) stopped at its first check, because the plain 67M already reads the sum's parity weakly (0.61 held out, bar 0.55); no fine-tune ran. Add the decision card "the sum-parity built control stopped" to pending decisions.
