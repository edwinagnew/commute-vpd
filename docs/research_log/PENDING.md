# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
A line `CONSENT YYYY-MM-DD <role> · <the user's words>` records that the user waived the main-page word cap for this flush (README.md → Main-page length; friction 688).


- 2026-11-17 definition-lead: P1.24 closed (E193 and E194, 2 of 2 runs, about $3.45 on RunPod H100s): calibration of a new candidate definition, causal-rule retention at a declared level (tested, not adopted), on three hand-built edits of the 67M at two levels (MLP neurons; heads plus MLP blocks). Do-nothing read retention 1 on all 12,288 neurons and 28 heads/blocks; a circuit of 8 layer-1 neurons with its output doubled read 0.80 and 0.91. The third edit, the same circuit replaced by different neurons computing the same values, was never built: in 7 fits across both runs, fresh neurons either copied a circuit neuron (max |corr| 0.996-0.9996) or failed to reproduce it (fit ratio 0.44-0.78). No level frozen; the five trained pairs not read. Premise check doubtful: what should the definition call new units that reproduce an old circuit? A decision card offers the next cell (recommended: build the replaced item by fine-tuning with the circuit knocked out). Page reports/analyses/2026-11-17_analysis_p1-24-rule-retention.md, figure experiments/rule_retention/figures/replaced_fits.png.
- 2026-11-17 definition-lead: reports/decisions/2026-11-17_decision_p1-24-next-cell.md · decisions (new top card, Question for you · definition-lead, id decision-p1-24-next-cell, urgency Soon, nothing runs until decided) · which one new cell should test causal-rule retention next: (a) fine-tune later layers with the circuit knocked out (recommended), (b) hard constraint, (c) accept moved copy, (d) read trained pairs on built items only; open question quoted with no recommendation.
