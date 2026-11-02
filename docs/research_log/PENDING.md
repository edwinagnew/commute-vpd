# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.

- 2026-11-02 · experiments/TRIED.md row "E12 Amendment 1", experiments/new_vs_existing/results/predictions_amend1.json · experiments (E12 card, one line) · coded and predictions filed, not run: the route split on teacher training's own states of E12's eight fine-tunes, plus-remainder read beside it.

- 2026-11-02 · decisions.md · card decision-after-e62-a3 (remove, decided), next, E62 card · the user chose the free test: E62 Amendment 4 is due to run, one cluster debug job, $0; does the copying fine-tune's mask open the pieces the original model's own mask opens where it copies, and does closing them remove the masked gain.
