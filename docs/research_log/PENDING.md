# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
- E69 (qwen-rl-lead, 2026-11-03): filed null prediction missed. Shuffled rewards gained 5.6 points (se 1.2) on GSM8K test under the drop rule against a filed |gain| < 1.5, and the filed early-signal ratio missed at step 100 (8.69 against 5.10, needed 3x). On capped answers the null reads +1.7 and -1.9 against RL +9.0 and +6.4; whole answers +4.9 and +3.0 against +9.4 and +7.8. Cause not established. Card when the arm comparison is read; report and notebook due.
