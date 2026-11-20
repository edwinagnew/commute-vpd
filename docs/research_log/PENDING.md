# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
A line `CONSENT YYYY-MM-DD <role> · <the user's words>` records that the user waived the main-page word cap for this flush (README.md → Main-page length; friction 688).
- 2026-11-20 · reports/analyses/2026-11-20_analysis_e230-topk-gate.md · E230 card (P2.47), built-pairs comparison table, timeline · VPD subcomponents with the gate function replaced: binary masks from its own importances keep 9 of 9 built pairs; a top-k gate on |x·V_c| at matched k keeps 6 (1 reversed, 2 level), above top-k on neurons (4); filed P1 "falls to the neuron level" as predicted; figure experiments/topk_gate/figures/shares_by_gate.png

- 2026-11-21 · reports/analyses/2026-11-21_analysis_e229-topk-upper.md · E229 card (new) and E217 card (add the upper-tail follow-up), neuron-featurization section, timeline · With the upper tail only (no GELU floor tail), base and fine-tune share about twice as many strong neurons, R rises toward full width but stays below it, and the built pairs' reuse-minus-twin R gaps shrink several-fold, with two pairs reversing at p99; figure experiments/topk_upper/figures/gaps_upper_vs_two_sided.png
- 2026-11-21 · reports/analyses/2026-11-21_analysis_e228-random-kl.md · E228 card (P2.44 filled), claims table (the KL-form share fails the random control on the 67M too; r has no random control), timeline · In Qwen's KL form, random changes matched on output change read 0.97-0.99 and reach 67 of 88 fine-tunes' own share, so a high share does not tell a fine-tune from a random change at 67M as on Qwen, while r refuses every random change; figure experiments/random_kl_share/figures/random_share.png
