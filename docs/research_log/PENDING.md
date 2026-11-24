# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
- 2026-11-24 · fleet/paper/cells.tsv (cell P2.76) · next · New cell P2.76 for regating-lead, 1 runs, due 2026-11-24: Figure 5 and the App. I difficulty figure standardised: every reading on the same tasks, gain read at matched states
- 2026-11-24 · fleet/paper/cells.tsv (cell P2.77) · next · New cell P2.77 for baseline-lead, 2 runs, due 2026-11-25: Figure 4 and App. I known_appendix standardised: 3 seeds a side on every known-answer pair, one task variant a side, one form of each reading
- 2026-11-24 · fleet/paper/cells.tsv (cell P2.78) · next · New cell P2.78 for regating-lead, 1 runs, due 2026-11-25: Realistic panel kept share at VPD: matched randoms for all three seeds
- 2026-11-24 · fleet/paper/cells.tsv (cell P2.79) · next · New cell P2.79 for compare-lead, 3 runs, due 2026-11-26: Reading refactor: every reading the paper uses gets one definition in src/commute/, each reproducing its saved numbers on $0 cluster debug jobs before any caller switches
- 2026-11-24 · decisions.md · decided · Paper framing: the definition's clauses named needed/kept; copy needs a base head but does not keep it; prior a confound only; claim 2 stated empirically (cheap readings follow how known fine-tunes were built and the mechanism over the prior, not a reading of the definition, realistic not validated). Plan reports/plans/2026-11-24_plan_results-framing.md
- 2026-11-24 · reports/analyses/2026-11-24_analysis_p2-76-equal-n.md (cell P2.76) · result · Fig 5 at equal n: on the 8 tasks all readings share, effective rank no longer reaches p < 0.05 with the prior partialled out (0.23, p 0.24; was 0.43, p 0.03 on 12), so no Fig 5 reading follows the tiers significantly; difficulty figure: old tasks cannot be KL-matched without retraining (1 of 36 in window), mismatch stated in the caption, steps-to-half dropped.
