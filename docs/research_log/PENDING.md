# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
A line `CONSENT YYYY-MM-DD <role> · <the user's words>` records that the user waived the main-page word cap for this flush (README.md → Main-page length; friction 688).
- 2026-11-12 · reports/analyses/2026-11-12_analysis_e163-qwen-rl-readable-run1.md · P3.4 card (Qwen RL readable), timeline · E163 run 1 (provisional): no RL recipe reached 5x the masked error with pass and fits held; best 3e-6 at 2.65x (step 150, pass 0.891); 1e-5 reached 5.23x at step 25 but fits 0.829; the measure checked on E69's saved update (0.0643, job 59301), the repeat is a different update of the same norm.
- 2026-11-12 · reports/decisions/2026-11-12_decision_qwen-rl-run2-recipe.md · pending decisions · new card: which recipe run 2 of P3.4 trains (recommend peak 3e-6 to step ~400, ~$170, over the cell's $200 RunPod allowance with run 1's $51; needs the user's yes and a /mcp reconnect for the plugin's $100 cap).
