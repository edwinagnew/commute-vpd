# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
A line `CONSENT YYYY-MM-DD <role> · <the user's words>` records that the user waived the main-page word cap for this flush (README.md → Main-page length; friction 688).
- 2026-11-11 definition-lead · P1.15 filled (E159, difficulty ladder): the rewiring share does not fall with difficulty. Rescaled between planted ceiling and floor it read 0.47, 0.07, 0.47 at one site (out of order, against the filed expectation) and 0.70, 0.59, 0.63 at two sites, following neither the plain model's MLP probe (1.0, 0.98, 0.72) nor its linear probe; the hardest rung's fine-tunes barely closed on their target in TV, so its rise is unexplained. Page reports/analyses/2026-11-11_analysis_difficulty-ladder.md; card reports/decisions/2026-11-11_decision_difficulty-ladder-result.md (Blocking).
- 2026-11-11 · reports/decisions/2026-11-11_decision_difficulty-ladder-result.md · decisions (pending-decision card, `Question for you · definition-lead · new 2026-11-11`, Blocking; nothing runs until decided) · New card: the difficulty ladder did not show the rewiring share falling with difficulty; options (a) retrain the sum-of-three rung and re-read all three on the same windows (recommended, $0, cluster debug, about 2 h, needs a new cell), (b) take the ladder as read and report within pairs only, (c) drop the graded reading.
