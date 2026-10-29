# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.

- 2026-10-29 · decisions.md (Pass criterion on Δ) · overview Result 2 headline, questions Q1 status, techniques table "Δ untouched" column and Status cells, E43 card ("stays the map") · "Δ untouched" is a description, not the criterion: the criterion is "Δ may not absorb δW", small changes to Δ allowed; E43's teacher training plus remainder is a live candidate (its faithfulness cost stated), not ruled out. Factual "with Δ untouched" in E41/E42 cards and claims stay.
- 2026-10-29 · reports/analyses/2026-10-29_analysis_e48-pass-reread.md (section "Amendment 1"), notebooks/E48_pass_reread.ipynb, experiments/TRIED.md (E48 Amendment 1) · E48 card, claims table (E42 "next: list the subcomponents carrying the gain"), timeline 2026-10-29 · E48 Amendment 1 (job 42960, $0): the carriers of E42's arm-A copies' gain at the cue number hundreds, n50 1,909 / 1,138 / 986 of 9,881 alive (rand32 / rand1 / ident32); P7 "tens" missed on both random ciphers; no small Γ change as the repair; the identity control spreads as widely, so no sign a new behaviour spreads more; figure e48_a1_carriers.png
