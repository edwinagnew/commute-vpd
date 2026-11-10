# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
- 2026-11-10 · reports/analyses/2026-11-10_analysis_e140-real-vs-twin.md · experiments (E140 card); table of work; timeline · P2.4 filled, provisional: per-word steering separates real fine-tunes from shuffled-label twins on both pairs (twin above real), pooled steering on the case pair by 0.101, the share does not, the neuron version is inert; likely causes are the per-word key and the twins' larger updates, untested; labels as built, no verdict
- 2026-11-10 decisions: remove the pending card "Every pair of fine-tunes we test on has already been looked at..." (ask 9, P2.7). User: no sealed pair; 67M is the validation set, Qwen the test set (decisions.md row "No sealed pair").
