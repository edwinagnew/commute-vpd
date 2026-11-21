# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
A line `CONSENT YYYY-MM-DD <role> · <the user's words>` records that the user waived the main-page word cap for this flush (README.md → Main-page length; friction 688).
- 2026-11-21 · sae-lead · reports/analyses/2026-11-21_analysis_e234-sae-train.md · E234 card (P1.40 run 1), timeline · base SAEs: FVE 0.875-0.995, recovered 0.926-0.999 (mlp2/mlp3 under 0.95), attn0-3 dead 0.53-0.62; 46 of 75 states drop Pile FVE > 0.02 somewhere (median worst 0.024), new_few/new_thin mlp0 task FVE drop 1.14-1.15, x10 resid_post3 16.2; 77M of 500M training tokens reread (rows ran out at step 206,464); figure experiments/sae_train/figures/fve_drop_by_site.png
- 2026-11-21 · sae-lead · reports/analyses/2026-11-21_analysis_e245-sae-r-a.md · E245 card (P1.40 run 2), claims table (R under wording A, SAE row), timeline · R_A at base-only attn/mlp SAE latents, bar 0.1 KL(F||M): reuse x2 0.9985, x10 0.852, rebuilt 0.694 (moved not read) above new_thin 0.571, new_few 0.516; gap 0.123 vs split-half band 0.110, rival a narrowly (VPD gap -0.120). Order rests on K* latents at untouched sites: over changed members only rebuilt 0.220 < new_thin 0.422, new_few 0.377. Remainder share does not separate (x2 0.894, new 0.775-0.792). 15 key items 0.989-0.999, cipher lowest; must-keep sets mostly site remainders, so at ceiling. Every K* search capped; no SAE random control; pairs not run. Figure experiments/sae_r_a/figures/r_a_sae_vs_vpd.png.
