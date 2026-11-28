# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
- 2026-11-27 definition-lead · P1.66 filled (E326): Fig 2 retained share on all 16 known-answer tasks x 3 seeds at VPD, top neurons and SAE; fine-tunes 0.011-0.059 (task means over seeds) vs matched randoms 0.003-0.018; reuse and new tasks overlap at every featurization; the neuron swap hook was fixed (identity 0.027/0.069 to at most 7.8e-5), rows moved by at most 3.8e-6; descriptive, no verdict · reports/analyses/2026-11-27_analysis_p1-66-fig2-all-tasks.md
