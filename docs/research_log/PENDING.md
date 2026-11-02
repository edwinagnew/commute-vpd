# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.

- 2026-11-02 · decisions.md · pending decisions (remove card decision-e55-c2-platform); E55 card Run line · user chose (a): C2 runs on the cluster normal QoS at $0 (Qwen-only rule relaxed for this run); gates before the run
- 2026-11-02 baseline-lead · decisions.md, reports/plans/2026-10-30_plan_baseline-next2-draft.md · Next box / timeline · The user approved E59: a Jain et al. per-block readout trained on Pile text, plus removal of its direction in the 15 saved E54/E56 fine-tunes, as a label for "half-known" independent of VPD; cluster debug, $0; coding now.
