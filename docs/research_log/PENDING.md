# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.

- 2026-11-02 regating-lead · decisions.md · pending decisions · Remove card decision-says-b: the user kept the filed rule; E62 Amendment 6 running.
- 2026-11-02 regating-lead · reports/analyses/2026-11-02_analysis_e62-amend6-overwrite.md · E62 card, claims table, timeline, pending decisions · Independently checked (reports/reviews/2026-11-02_review_e62-amend6-analysis.md): E62 Amendment 6 reads filed `mixed`, worded "leans to B": with B overwritten under W0's held mask at the second A, about 0.65 of the lean to B survives, B's top-1 falls 0.195 to 0.035, the new token gains 0.053 of the lean; read on the new text, the mask favours the new token; neither of the user's two branches fires, and the numbers go to the user. Arithmetic after the run, no filed rule, assuming the two tokens are exchangeable: neither account alone fits; about 0.62 needs the mask and not the token in the text, 0.32 needs both, 0.06 the text alone; it does not name what carries either part.
