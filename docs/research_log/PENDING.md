# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
- 2026-11-04 · reports/analyses/2026-11-03_analysis_e69-grpo-loop.md · experiments card E69, claims table, timeline, new pending-decision card · E69 Amendment 1, provisional until an independent check: RL's lead over the shuffled-reward null on GSM8K test is +3.2 strict and +1.25 format-relaxed on seed 1, +6.42 (1.07) on both on seed 2, the filed lean "ambiguous or no lead" reversed, the seed spread of about 5 points unexplained, the null's gain +1.62 (1.11) just past its band. E69's card's "What we found" paragraph still says "the firm lead may be largely answer format" without limiting that to seed 1: rewrite that sentence. The card text for the third-seed decision (about $20; recommendation read the two first) is in the report under "Pending decision".
