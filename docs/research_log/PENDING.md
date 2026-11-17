# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
A line `CONSENT YYYY-MM-DD <role> · <the user's words>` records that the user waived the main-page word cap for this flush (README.md → Main-page length; friction 688).
- 2026-11-17 · reports/analyses/2026-11-17_analysis_share-built-kind.md · new E200/E202 card / P2.33 status (both runs done) / timeline · The re-gating share read on E139's built-kind fine-tunes, on windows where the hand-made edit moves its top token by 1+ log-odds: Table about a quarter, Always on, Cue and XOR (learned nothing) all a little above one; Always on beat Table but fell below Cue, so the share does not separate built re-weighting from rewiring here, and its one split follows gain size, untested with one large-gain group; provisional, no independent check.
