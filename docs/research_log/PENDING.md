# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
- 2026-11-04 · reports/reviews/2026-11-04_review_e85-adversary-check.md · E85 card, decision-after-e76 · independent check: filed outcome computes, no code bug, but 'cannot fail a bad state' not supported: no positive control, random change is not bad in the sense the test reads, rarely-on is bad only under frozen masks. Evidence wording on the card to change; ask 27 replaced by the new ask.
