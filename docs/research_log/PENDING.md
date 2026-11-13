# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
A line `CONSENT YYYY-MM-DD <role> · <the user's words>` records that the user waived the main-page word cap for this flush (README.md → Main-page length; friction 688).
- 2026-11-13 · decisions.md · pending decisions (remove card decision-qwen-share-reference), timeline · The user chose option (a) on the Qwen share reference card ("yes, run option a"); cell P3.10 added (1 run, due 11-16).
- 2026-11-13 · qwen-rl-lead · timeline, P3.4 card · Cell P3.4 (RL on Qwen against matched nulls) closed after its 3 runs with the verdict "no call": the shuffled-reward null collapsed at both learning rates, so no size was read against a working null. The premise check read "doubtful": the null may not be steerable to RL's size without collapsing, and every update, RL's included, sits at 0.98-0.99 of a random change's outside share. Report reports/analyses/2026-11-13_analysis_e178-qwen-rl-readable-run3.md.
- 2026-11-13 · qwen-lead · timeline, P3.10 card · Cell P3.10 (a reference for the share on the frozen-neuron Qwen decomposition) ran once, E181: the filed rule makes no call, as only 7 of 15 switched-off neuron sets landed at their target size (9 needed); 5 of those 7 read above random changes of the same size and 2 inside. The library's fine-tunes split: above the random band on gold and known rows, inside on copy rows. Page reports/analyses/2026-11-13_analysis_e181-qwen-share-reference.md.
- 2026-11-13 · definition-lead · pending decisions (add card decision-e182-controls-stop), timeline · The ranking test of the fourth wording stopped on its controls run: one built control read as built, the other (a neuron triggered by a principal direction) read rung 3 because the search took no step; decision card asks whether to close "no call" or fix the search and rerun the controls.
