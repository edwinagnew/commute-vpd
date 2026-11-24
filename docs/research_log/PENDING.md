# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
A line `CONSENT YYYY-MM-DD <role> · <the user's words>` records that the user waived the main-page word cap for this flush (README.md → Main-page length; friction 688).
- 2026-11-23 · reports/analyses/2026-11-23_analysis_p2-67-adl-published.md · P2.67 card (filled), E284 card, E264-267 cards (superseded), ADL row in the claims/baselines table · The activation difference lens re-run as published with the authors' toolkit: on the paper's own Qwen3-1.7B fine-tunes it passes every filed rule (training-text rise 0.027 nats vs random bar 0.0013; mixed-in 0.53 of plain), but on the 67M its filed control (lowercase) fails, the lens's direction below the random mean, so none of our states was read. E264-267's score is not in the paper; the paper detects the training domain and makes no reuse call, which is the methods-table label. Port vs toolkit mean difference mismatch unexplained (on Not run).
