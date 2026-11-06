# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
- 2026-11-06 · reports/analyses/2026-11-06_analysis_e122-damage-crossed-pair.md (notebook notebooks/E122_damage_crossed.ipynb, figures experiments/damage_crossed/figures/) · new card e122 (provisional, outcome "no answer"); E39 card (gap a open: crossed pair gave no call); claims table (broad-fragility cause); timeline 2026-11-06; next/gaps (split-start 8.33 to run down) · E122 damage-vs-recipe pair: Adam at LoRA damage 2.84/1.44 (mixed), LoRA at Adam damage 1.53/1.60 (between, copy off band) → no call, lean only that both copies move with damage; copy of the full Adam fine-tune from its least-squares split fails far worse (start 8.33, copy 9.40/9.82 vs standard 2.17/1.90) though it fits the fine-tuned model as well; LoRA copies keep adversary excess at capital-next-token positions, Adam copies spread it
