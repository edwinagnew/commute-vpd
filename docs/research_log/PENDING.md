# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
- 2026-11-06 · reports/analyses/2026-11-06_analysis_e103-known-answer-pair.md, notebooks/E103_known_answer_pair.ipynb, experiments/known_answer_pair/figures/{gain_curve,w0_deficit,recall,pair_rules,prompt_overlap}.png, experiments/TRIED.md (E103 row) · experiments page (E103 card: coded → ran, provisional, chip "surprise"; figures gain_curve, pair_rules), What's next (neuron baseline line), timeline (2026-11-06) · E103, the known-answer pair, ran: every neuron reading level at matched size as filed, but the gain control (11.35 nats a token against a bar of 0.33) passed mostly on a format habit, the model learning to stop after a bare short answer, so the matched pair differs in answer format as well as label and its level readings do not yet count against the instruments; end states separate on prompt overlap with a 2x size difference (reported); next a $0 per-position gain read without the end token (provisional).
