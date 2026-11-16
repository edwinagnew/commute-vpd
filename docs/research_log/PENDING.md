# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
A line `CONSENT YYYY-MM-DD <role> · <the user's words>` records that the user waived the main-page word cap for this flush (README.md → Main-page length; friction 688).
- 2026-11-16 · qwen-lead · P3.10 card, timeline · Cell P3.10 filled (E181, page reports/analyses/2026-11-13_analysis_e181-qwen-share-reference.md): controls held; the filed rule made no call (7 of 15 built switch-offs scorable, needs 9; 5 above the random band, 2 inside); 25 of 36 fine-tunes above the band, gold and known above, copy inside. Premise check "doubtful": fine-tunes may simply be more structured than random changes, so the random curve may not single out re-gating; listed for the 11-18 gate.
