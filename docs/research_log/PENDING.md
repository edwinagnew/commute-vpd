# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
A line `CONSENT YYYY-MM-DD <role> · <the user's words>` records that the user waived the main-page word cap for this flush (README.md → Main-page length; friction 688).
- 2026-11-21 definition-lead · reports/analyses/2026-11-21_analysis_e257-eb-trained.md, experiments/rb_states/results/summary.json, experiments/rb_states/figures/eb_arms.png · E257 card (provisional to final), claims table, timeline · E257 final, all three stages read, descriptive: 94 SFT fine-tunes E_b 0.015-0.062, new side higher in 6 of 9 pairs; repeat RL 0.17 (size-matched) and 0.11 (KL-matched) vs own/demo SFT 0.016-0.021, shuffled 0.037; procedural tasks (addition, brackets, sequence) every trained arm 0.037-0.075, RL within 1.5x of SFT, matched randoms 0.0092-0.013; all 66 fresh K* searches hit the 2,000 cap (a limit). RL figures now from the report and summary.json, not the decision card; add figure eb_arms.png; drop "preliminary".
