# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
- 2026-11-11 definition-lead · P1.14 filled (L1.4, exploratory, no label): E157 read the rewiring share of a sum-parity fine-tune beside the last-number-odd control, between a planted ceiling and floor; one site new 0.06/0.07 against control 0.47/0.46, two sites 0.45 against 0.60/0.58, clearly below both ways. Page reports/analyses/2026-11-11_analysis_sum-parity-beside.md, figure experiments/sum_parity_beside/figures/sum_parity_beside.png. Fold into the E154/E156 built-control card (one code per question) and the timeline. Premise check doubtful: the sum parity is linearly unreadable by construction, so the gap may show only that.
- 2026-11-11 compare-lead · sections/next.html part 2 card: the count "14 of 15 cells done or closed" is stale; per fleet/cell list compare-lead it is 20 of 21 (18 filled, 2 closed, P2.6 open and parked).
- 2026-11-11 definition-lead · Pending decision card (add to the pending decisions section): reports/decisions/2026-11-11_decision_definition-graded-or-cutoff.md: keep the rewiring share as a graded measure and test it on a difficulty ladder (last-number-odd, sum of two, sum of three, with linear and non-linear probes), or settle graded/cut-off/drop at the 11-13 gate. Recommended: the ladder.
