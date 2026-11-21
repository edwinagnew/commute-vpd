# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
A line `CONSENT YYYY-MM-DD <role> · <the user's words>` records that the user waived the main-page word cap for this flush (README.md → Main-page length; friction 688).
- 2026-11-20 · decisions.md · definition box, decisions, timeline · Definition wording A adopted (user ~19:50): kept = M mechanism, up to an output gain, can stand in for the variable and the behaviour survives (soft intervention); a memorised fact reads new where stored however small the change; R redone under A on the six built edits and nine pairs at matched size only, other R readings become future work.
- 2026-11-21 · reports/analyses/2026-11-21_analysis_e235-crosscoder-proper.md · E235 card (new, under the baselines), claims table baseline row, timeline · The authors' crosscoder run properly (their trainer, TF32, full tokens, 3 seeds): block 2 misses its planted control (grafted head 0.30/0.21/0.37 vs 0.5); block 3 passes it only on a head that raises Pile loss 2.71 to 11.30 nats, and there a matched random update reads like both e54 sides (0.0035-0.0070 vs elicit 0.0056-0.0092, cipher 0.0014-0.0106): not applicable at this update size at both sites; run 2 = four layer-3 grafts at block 3 only.
