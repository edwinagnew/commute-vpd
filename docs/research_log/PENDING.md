# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
- 2026-11-07 regating-lead · reports/analyses/2026-11-07_analysis_e130-copy-protection.md · E130 card (new) + timeline · E130 (job 55000, $0, provisional): teacher training's placement protection is not a constant of the recipe; the copy of the Adam fine-tune stopped at step 20 reads 2.84 placed and 2.17 with its change in Δ (m 1.58, placement adds excess), the step-2,400 LoRA copy 1.53 vs 3.17 (m 0.25); over four lowercase copies m 0.24-1.58, one seed each; headline "function-dependent" as leaned; figure experiments/copy_protection/figures/placement.png
- 2026-11-07 regating-lead · reports/analyses/2026-11-07_analysis_67m-story.md · claims section + timeline · update of the already-linked 67M synthesis with E130: teacher training carries narrow fine-tunes with named exceptions and fails the broad lowercase Adam fine-tune; its placement hides most of that copy's excess but varies by copy (m 0.24-1.58 over four lowercase copies, adds excess on the step-20 Adam copy); no 67M case where the two readings of reuse agree on a positive; the "Teacher training carries the broad full-matrix lowercase fine-tune" claims row should note placement is not protective on every copy
