# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
A line `CONSENT YYYY-MM-DD <role> · <the user's words>` records that the user waived the main-page word cap for this flush (README.md → Main-page length; friction 688).

- 2026-11-11 · reports/decisions/2026-11-11_decision_qwen-overlap-floor.md · decisions (new card, first in section; `<div class="k">` Question for you · compare-lead · new 2026-11-11; urgency chip Blocking, line "Blocking for P3.6 run 2 only"; due 11-13), changelog · the random update's neuron overlap on Qwen moves more between two draws than the filed tolerance (draw differences 7.5e-5 and 0.0023, about twice the allowed 3.6e-5 and 0.0015); options a four draws (recommended), b two draws as filed, c close on effective rank only; FYI in the card: all 19 updates have lower effective rank than their random floor (1057-1443 vs 1812-1825)
