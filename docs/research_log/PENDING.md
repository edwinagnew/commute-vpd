# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
A line `CONSENT YYYY-MM-DD <role> · <the user's words>` records that the user waived the main-page word cap for this flush (README.md → Main-page length; friction 688).
- 2026-11-20 · decisions.md · definition box (readings), decisions · Measurement change, not wording: R under wording A (P1.46/E242) reads 'behaviour preserved' with a relative bar, KL(F || F_S) <= 0.1 KL(F || M); E226's absolute allowance let the do-nothing, random controls and new_few read R = 1.
- 2026-11-20 · sae-lead · E234 card (P1.40 run 1): base-only SAEs at all 13 sites of the 67M; base FVE 0.875-0.995, recovered 0.926-0.999, dead 0.53-0.62 at attn0-3 (a miss); 46 of 75 states drop FVE > 0.02 somewhere; x10 resid_post3 outlier (drop 16): 1% of positions hold 99.6% of the error, a real reading of 10x neurons, not a bug. TRIED.md row E234.
- 2026-11-20 · reports/analyses/2026-11-20_analysis_e241-gating-vpd.md · E241 card (P1.45), definition box (readings), timeline · Gating at VPD (M's VPD subcomponents, gates from F's activations) does not separate reuse from new on the built edits: S not read (reuse edits under VPD's floor), S_c rival b (gap -0.128, band 0.158), random control fails (0.56-1.87 vs reuse 0.61-0.69); trained pairs reuse above new 8/8 on S, 9/9 on S_c, descriptive. No claim to writeup-lead. TRIED.md row E241.
- 2026-11-21 · reports/analyses/2026-11-21_analysis_e246-realtext-proxies.md · E246 card (P2.53), claims table, timeline · No cheap proxy orders the four real-text pairs by their built labels: prior 3 of 4 hits (British circular, language reversed), log-odds share 1 of 3 readable, P1.45's S 1 of 4 (two pairs under its floor; unvalidated per E241); do-nothing exactly 0, random gains at most 0.065. Part 2's proxy claim stays on built items. TRIED.md row E246.
