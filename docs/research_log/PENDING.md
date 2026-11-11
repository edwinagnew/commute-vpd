# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
A line `CONSENT YYYY-MM-DD <role> · <the user's words>` records that the user waived the main-page word cap for this flush (README.md → Main-page length; friction 688).
- 2026-11-11 compare-lead · P2.6 filled (E161, job 59106, $0; un-parked by the user 11-11): the every-position share on the code-text fine-tune that made held-out text worse is stable across five row batches, 0.78 at step 10 and 0.80 at step 100 (batch sd about 0.03). A random update of the same size in every matrix lowers the text about 39 times less and gives no readable share, so size alone does not produce the 0.8; a random update that lowers the text as much was not run. Page reports/analyses/2026-11-11_analysis_p2-6-share-stability.md, figure experiments/share_stability/figures/stability.png.
- 2026-11-11 compare-lead · P2.22 filled (part 2 gate page for 11-18, 0 runs): the scorecard redrawn with every result to date (figure experiments/method_table/figures/gate_scorecard.png). The share alone puts the reuse side higher on all nine built pairs, but the prior and strength-1 steering together make the same nine calls, and the pair where they disagree could not be built. A proposal for the gate lists which methods go to Qwen. Page reports/analyses/2026-11-11_analysis_part2-gate-page.md.
