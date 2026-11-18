# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
A line `CONSENT YYYY-MM-DD <role> · <the user's words>` records that the user waived the main-page word cap for this flush (README.md → Main-page length; friction 688).
- 2026-11-18 · decisions.md · pending decisions, what is next · remove the card decision-p2-31-crosscoder-miss (answered by the user 2026-11-18: implement exactly as Minder et al., replicate a result, then realistic tasks); add rows P2.37 (crosscoder check against the authors released Gemma-2-2B crosscoder) and P2.38 (realistic task set on the 67M) as open
- 2026-11-18 · redteam-lead · P1.32 filled · reports/reviews/2026-11-18_review_definition-cold-read.md · cold read of the working definition under both answers to ask 48: (a) as written reads an old skill aimed at a new cue as new and gives no reading when the base model already has the behavior; (b) reads reward hacking and emergent misalignment as mostly reused but a memorised fact as mostly reused too; neither names "created" or compares two pretraining runs; 11 clauses no reading uses could move to an appendix · add a line to the definition section linking the page
