# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
A line `CONSENT YYYY-MM-DD <role> · <the user's words>` records that the user waived the main-page word cap for this flush (README.md → Main-page length; friction 688).
- 2026-11-23 P2.68 filled (E288): head-level CMAP (Prakash 2024) makes no call on the 9 built pairs (reuse above new on 5, 6 needed); the older E248 reading R_A (not the paper's ρ) separates all 9; the top/cipher reversals are circuit size (path patching keeps 1-2 heads there; level against same-size random head sets). Page reports/analyses/2026-11-24_analysis_p2-68-cmap.md.
- 2026-11-24 writeup-lead: paper v2 (pushed 62494d3) Results rebuilt to the user's 11-23 order: 5.1 the definition read at VPD, top neurons and SAE on one figure (copy head as the worked case); 5.2 fine-tunes with a known answer (copy-heads-off label check, re-gating, steering, effective rank, base prior; claim 2 holds there); 5.3 realistic fine-tunes, same readings, not validated; 5.4 existing tools as one table (crosscoder rescored per P2.70). Appendix cut from 28 to 10 pages, cut text kept in paper/cut/.
