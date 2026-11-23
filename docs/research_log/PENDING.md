# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
A line `CONSENT YYYY-MM-DD <role> · <the user's words>` records that the user waived the main-page word cap for this flush (README.md → Main-page length; friction 688).
- 2026-11-23 · experiments/convergence_panel/results/summary.json (wide.base_loss) · P2.66 card (post-hoc, descriptive) · E272 data-only check: how unfamiliar each task's training text is to the base model (its loss per byte on the trained tokens, job 64181) tracks the blind tiers moderately (tau 0.51 on the 8 continued-text tasks, 0.46 on all 12). Within the 8 text tasks effective rank still follows the tiers after removing it (partial tau 0.87, p 0.0008), so there erank carries more than text unfamiliarity; across all 12 it does not (partial 0.20, p 0.20). Filed before computing; chosen after the first results, so descriptive.
