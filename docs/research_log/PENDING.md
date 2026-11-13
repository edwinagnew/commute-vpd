# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
A line `CONSENT YYYY-MM-DD <role> · <the user's words>` records that the user waived the main-page word cap for this flush (README.md → Main-page length; friction 688).
- 2026-11-13 · reports/decisions/2026-11-13_decision_qwen-share-reference.md · pending decisions (new card), timeline · Decision card for the user (qwen-lead, following the 11-12 direction on Qwen SFT decomposition / re-gating): does the re-gating share read a built re-gating change (neurons switched off) as re-gating on the frozen-neuron decomposition, with random changes at three sizes as reference, before part 3 relies on it? Recommended (a), one run, ~$12; (b) more random draws; (c) where SFT updates land among neurons.
