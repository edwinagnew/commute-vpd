# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
- 2026-11-11 · reports/analyses/2026-11-11_analysis_built-control-xor2.md · E148 card (built control), claims table, sub-question status of the working definition, timeline · Built control on two-key XOR: both arms learned the target but the rewiring share stayed far under the 0.5 cut-off on both (taught original 0.02, plain 67M 0.08 and 0.09), no map found on either; by the user's condition the wording needs revising, but the taught original held the skill only weakly (linear read 0.588, chance 0.50), so whether the search fails in general is open. Figure experiments/built_control/figures/h_by_arm_scored.png.
- 2026-11-11 · reports/decisions/2026-11-11_decision_built-control-result.md · section 3 pending decisions (new card) · Pending decision: revise the reuse wording now, or first run one $0 cluster run with a strongly taught original (linear read above 0.9) to see whether the search fails in general or only on weakly held variables, or keep the wording and report built as unreadable; recommendation: the strongly taught run.
- 2026-11-11 definition-lead: remove the pending-decision card for ask 31 (built-control-result); user chose neither: keep the wording, find a more natural built control.
