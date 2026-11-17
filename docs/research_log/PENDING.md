# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
A line `CONSENT YYYY-MM-DD <role> · <the user's words>` records that the user waived the main-page word cap for this flush (README.md → Main-page length; friction 688).


- 2026-11-17 definition-lead: P1.23 filled (E192, 1 of 2 runs, $7.75 on a RunPod H100): filed test of the curve reading on fresh seeds 2 and 3 of the five built pairs, same 16 names read twice. The rule (reuse side closes more than new by over 0.0295, the largest E184 same-side seed gap, on every pair and seed) fails on both readings: full next-token 7 of 10, answer tokens 8 of 10 (the answer reading was chosen after seeing E184). Copy/cipher, identity/random map and parity twin/fact separate on every seed (gaps 0.17-0.50); repeat never does; case only narrowly. Controls passed; all 20 learned. Matches filed rival B, not a reversal. Fourth wording tested, not adopted. Premise check doubtful: are case and repeat really reuse against new? Page reports/analyses/2026-11-17_analysis_e192-curve-test.md, figure experiments/curve_test/figures/gaps.png.
