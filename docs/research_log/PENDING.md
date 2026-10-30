# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.

- 2026-10-30 qwen-decomp-lead: E57 A-C · reports/analyses/2026-10-30_analysis_e57-reading-the-update.md, notebooks/E57_qwen_reading.ipynb, experiments/qwen_reading/results/boot_direct.json, figures experiments/qwen_reading/figures/{regating_share,direct_share,switched_on,overlap_excess,placement,where_kind}.png, experiments/TRIED.md · E57 card, section 3 pending decision card (E57 no teeth: the bootstrap answers its "what would change it" line), claims table, timeline · Stages A-C written up: re-gating share 0.99 for every change incl. both random ones (no teeth at scale); direct share row bootstrap: gold 0.092 [0.090, 0.093], E40 0.063, random 0.022 and 0.029 [0.020-0.032], anchor 2 0.012 below random; gold and random intervals do not overlap, so "close D3 now" is not triggered; switched-on sets all under the filed floor and overlaps 19-76 sd for random too (floor and null wrong as filed); no prediction scored (anchor 1 never matched, §11); Stage D still waits on the user
