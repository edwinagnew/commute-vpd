# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.

- 2026-11-02 · docs/research_log/parts/sections/techniques.html · techniques (route split), changelog · notation only, no number or claim changed: the user's draft replaces v178's ℓ form. Bare symbols stand for log-odds, δW = W1 − W0 is the gain in this subsection, C_{a,b} is C_a gated by g_b, one bullet per share; the changelog row says reports, code and results files keep T, U, L, G.

- 2026-11-02 · experiments/TRIED.md row "E12 Amendment 2", experiments/new_vs_existing/results/predictions_amend2.json · experiments (E12 card, one line) · coded and predictions filed, not run: a free debug job on why the re-switching term falls on rand32b's students, with a second batch order, set overlap at the trigger and which values give the fall back.
