# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.

- 2026-11-02 · reports/plans/2026-11-02_plan_e67-weight-side-read.md · next (`next.html`) · E67 changed after review: do-nothing side only, null also read on the rows r failed on; not run.
- 2026-11-02 · reports/reviews/2026-11-02_review_e34-amend9-analysis.md · claims and E34 card · wording fixes after the check: claims row 58 "moves as far along the update as taught ones" -> "reads as high along the update as taught ones"; e34.html line 32 "a fine-tune that taught nothing" -> "a fine-tune that teaches no task"; same row keeps "teach-nothing" only if defined as teaching no task.
