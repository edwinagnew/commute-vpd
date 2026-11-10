# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
- 2026-11-10 · decisions.md · Pending decisions; decisions · remove card decision-finetune-imitators-table: the user chose option a, the Table items run as a stage of E139 rerun 2
- 2026-11-10 · reports/analyses/2026-11-10_analysis_e143-holes-fill.md; docs/research_log/parts/experiments/e143.html; parts/sections/methodtable.html; notebooks/P2_compare_methods.ipynb section 7; README.md figure list · experiments (new E143 card); methodtable (E143 paragraph added); table of work · P2.10 filled, provisional: on the seven saved built pairs the share separates every pair, fair-form steering two in its direction and three the other way, per-key vectors read about the whole gain on both sides, the neuron version is inert; labels as built, no verdict; publish needs fig/e143_gaps.png ("experiments/holes_fill/figures/gaps.png") in the files map
