# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.

- 2026-11-02 · reports/analyses/2026-11-02_analysis_e12-amend3-fp32-read.md, notebooks/E12_new_vs_existing.ipynb, experiments/TRIED.md row "E12 Amendment 3" · experiments (E12 card, one clause, plus report and notebook links) · add the clause: the fall on rand32b's students does not depend on the read's precision (read in bf16 and fp32, no fall moved by more than 0.0064 nats); this rules out read precision only and the cause stays open.
