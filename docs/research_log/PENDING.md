# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.

- 2026-11-03 · decisions.md; reports/analyses/2026-11-03_analysis_qwen-line-story.md · section 3, overview Line 3, Q9, next steps, timeline · Remove card decision-e61-next (user chose option 1, stop and write up); overview and next steps say the Qwen line has stopped and link the whole-story write-up.
- 2026-11-03 · reports/plans/2026-11-03_plan_qwen-reorient.md; decisions.md · decisions (cards), next, timeline, overview Line 3 · Remove cards decision-e61-next (superseded by the reorientation), decision-benchmark-next (option (a), the 67M question stays unanswered), decision-chain-size and decision-e47-split-keep (option (a) each, wording only) and the E62 "still says B" follow-up (E62 closed as the limit of the copying-mask work); "What's next" opens with the three Qwen deliverables (D1 map with a test that can fail, D2 SFT and RL, D3 elicitation against new) and the eight-day table, the 67M lines marked stopped; timeline line "2026-11-03 fleet reoriented to the Qwen deliverables; fast lane adopted"; the line above about the Qwen line having stopped is superseded (D1 resumes, the whole-story write-up stays linked if it exists).
