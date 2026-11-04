# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
- 2026-11-03 · reports/analyses/2026-11-03_analysis_e75-rl-reads.md · experiments card E75, claims table, sub-question status, timeline · E75's answer replaced, from `notebooks/E75_rl_reads.ipynb`, new figures `bad_placements.png`, `r_vs_n.png`, row in TRIED.md: three wrong changes at the update's Frobenius norm per matrix all fail on all five updates (narrowest another update's on the template fine-tune, gap 0.023 against half-width 0.014), the table still barely separates same-size parts and input-weighted size is untested; the unruled lean that another update's reads near the ruler missed (0.54, 0.63 on the first and gold fine-tunes), no read tests why; the shuffled-reward null reads every filed label RL reads, masked rows included, so those labels do not separate them; RL's update not called learning; own-samples arm queued as job 50874.
- 2026-11-03 · reports/analyses/2026-11-03_analysis_e78-anchor1.md, notebooks/E78_qwen_anchor1.ipynb · experiments card E78, timeline · ran: anchor 1 retrained at E61's rate reproduces E61 (step 100 within 1.7e-6, all 256 answers byte-identical, template 92.6%, strict accuracy 62.1%, `learned` true), with fp32 δW at steps 1, 2, 3, 5, 10, 20, 50, 100 ready for the split readings; step t has t − 1 moving steps, so later labels are one moving step behind E69's.
