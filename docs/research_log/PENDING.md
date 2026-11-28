# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
- 2026-11-27 · reports/analyses/2026-11-27_analysis_p290-share-unified.md (cell P2.90) · filled · One re-gating share on both models: the whole-answer log-odds share reproduces the 67M paper values within 0.002 on 47 of 48 states; the Qwen appendix re-read in it reads VPD 0.57-0.74; against random changes matched to an answer gain of 1, the fine-tunes read higher on cipher, invented and the maths tasks, and the randoms read as high on copy and known
- 2026-11-27 · fleet/paper/cells.tsv (cell P2.91) · next · New cell P2.91 for regating-lead, 1 runs, due 2026-11-28: Does the Qwen re-gating share depend on answer gain? Fine-tunes scaled down to gains 1 and 10, and a repeat read of VPD known draw 2
- 2026-11-27 · decisions.md (Qwen share gain test) · remove card decision-qwen-share-gain · user ~18:1x: "go with your recommendation on regating" -> option (a), cell P2.91 opened for regating-lead
- 2026-11-27 · decisions.md (new vs unexplained) · remove card decision-new-vs-unexplained · user ~18:1x: "this has always been a risk. i dont think we can convincly say. i ahve a feeling this is why qwen failed" -> option (a): stated as a limit in the draft's discussion, no run; the user's hunch that it explains the Qwen result is recorded as a hunch, not a finding
