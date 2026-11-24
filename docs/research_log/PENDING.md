# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
A line `CONSENT YYYY-MM-DD <role> · <the user's words>` records that the user waived the main-page word cap for this flush (README.md → Main-page length; friction 688).
- 2026-11-23 · Pending decisions (new card) · definition-lead · Proposal 64: a $0 synthesis from saved arrays of why the needed-set readings fail on trained fine-tunes, for the Discussion (user via whip-lead 11-23 ~17:20). It reads, across every saved E_b item at VPD and neurons/heads: whether the most-needed variables are rotated rather than rescaled (built x2/x10 as contrast), how much the needed set restricts (|K| over changed variables, margins), where the change sits (remainder share, per layer), and whether toy trained pairs fail like the realistic panel. It tells us whether the failure is general to trained updates and not task choice or code. Recommendation: approve. Line for the user: ~/commute-agents/fleet/cell approve 64
