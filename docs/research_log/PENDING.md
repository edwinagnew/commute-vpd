# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.


- 2026-11-02 · experiments/TRIED.md row "E12 Amendment 2", experiments/new_vs_existing/results/predictions_amend2.json · experiments (E12 card, one line) · coded and predictions filed, not run: a free debug job on why the re-switching term falls on rand32b's students, with a second batch order, set overlap at the trigger and which values give the fall back.
- 2026-11-02 · reports/reviews/2026-11-02_review_e62-amend4-check.md, reports/analyses/2026-11-02_analysis_e62-amend4-control-stop.md, notebooks/E62_induction_cue.ipynb · experiments (E62 card), figure title · independent check of E62 Amendment 4 done: verdict `control_failed_mask` and every number stand. Corrected reading: at the second A the target model's log-odds of B fall 4.82 with head (2,4) off and how much of that is copying is unread; the masked forward falls 0.92 with the mask held, and its lean of 4.34 has each set of tokens under its own mask, so the two are not like for like. Next is Amendment 5, being coded, not run.
- 2026-11-02 · docs/research_log/parts/sections/techniques.html · techniques (route split), changelog · rewritten for readability at the user's request, no number changed: says plainly that the shares split a log-odds gain and not the weights; scores now have their own letters, F (full), S (subcomponents), M (masked), which are T, U and L in reports, code and results files, and W, C and g stay the weights, subcomponents and gates; the gate and trigger preamble is cut.
