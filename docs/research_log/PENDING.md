# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.

- 2026-11-02 · reports/analyses/2026-11-02_analysis_e61-recipe-checks.md · E61 card, claims table, timeline · E61 card needs Job S (line C stands, resting on step 8 alone; big gaps 0.038 and 0.045; figure `experiments/qwen_recipe/figures/log_size.png`), the facts arm (invented facts at 6e-6, recall 0.987) and the rehearsal; steps 10, 20, 50 are exploration, no result yet.

