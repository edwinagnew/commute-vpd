# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
- 2026-11-07 qwen-decomp-lead · reports/analyses/2026-11-07_analysis_e131-new-decomposition.md · new E131 card + claims table (Qwen apparatus row) + timeline · E131 (MATS debug L40 jobs 55508-55542, $0, port controls pass): Goodfire's second Qwen3-1.7B decomposition p-9707d6ff against p-e9fcffd5 on 128 rows each of web text and GSM8K answers (thinking off/on); under the adversary (one seed) it reads 0.14-0.15 of the old KL on maths (11.4 → 1.6, 6.6 → 1.0) and 0.75 on web text; with its own masks 1.07 (thinking off, filed prediction missed) and 0.88 (thinking on) of the old, 1.48 on web text; 26-30% fewer active subcomponents. Provisional; the new decomposition looks the better base for GSM8K readings (pending card for the user).
