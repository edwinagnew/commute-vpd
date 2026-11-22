# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
A line `CONSENT YYYY-MM-DD <role> · <the user's words>` records that the user waived the main-page word cap for this flush (README.md → Main-page length; friction 688).
- 2026-11-21 definition-lead · E236 claims row: change the chip from "refuted" to "mixed" (at VPD the strength-only fine-tunes read lower on 49 of 49 pairs, but per head and neuron they read higher on 57 of 57; the card's stop rule was not triggered) · reports/analyses/2026-11-21_analysis_p1-41-gain-only-r.md
- 2026-11-22 sae-lead · reports/analyses/2026-11-21_analysis_e245-sae-r-a.md · E245 card, claims row (SAE row of R), timeline · E187's nine pairs at SAE latents: reuse side higher in 9 of 9 but by 0.0001-0.030, all inside the band 0.110, where VPD (E248, same 76 fine-tunes) separates all nine by 0.11-0.32; both sides read 0.96 or more; the remainder share splits the sides seed by seed (reuse 0.54-0.85, new 0.34-0.50); new figure pair_gaps.png (SAE beside VPD; no neuron reading under wording A exists)
- 2026-11-22 sae-lead · reports/analyses/2026-11-22_analysis_e251-sae-random.md · E251 card, E245 card, timeline · random updates at SAE latents read 0.48-0.54 against key fine-tunes 0.989 or more (gap 0.451, rival a as filed), but every random search hit its cap (lower bounds) and the random updates move the output over a thousand times less than the fine-tunes, so near-one is not automatic yet not shown to mean reuse
- 2026-11-21 definition-lead · parts/sections/decisions.html · decisions · remove the card decision-wording-a-at-vpd (ask 58 answered: user 2026-11-21 ~19:20 chose (b), count only what changed, each changed part weighted by the change it carries; decisions.md row 'Option (b) adopted for reading reuse')
