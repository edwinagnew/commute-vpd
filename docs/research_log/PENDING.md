# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
- 2026-11-04 · reports/reviews/2026-11-04_review_e85-adversary-check.md · E85 card, decision-after-e76 · independent check: filed outcome computes, no code bug, but 'cannot fail a bad state' not supported: no positive control, random change is not bad in the sense the test reads, rarely-on is bad only under frozen masks. Evidence wording on the card to change; ask 27 replaced by the new ask.
- 2026-11-04 · reports/plans/2026-11-04_plan_e85-adversary-teeth.md · E85 card, experiments page · E85 Amendments 1 (positive controls: random change times 30, then permuted columns) and 2 (does Γ rounding V to bf16 matter, $0) coded and filed, not run; permuting all alive columns is trivially failed, the rarely-on half coded as the fix.
