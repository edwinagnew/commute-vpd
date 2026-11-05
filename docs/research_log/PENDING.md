# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
- 2026-11-05 · reports/analyses/2026-11-05_analysis_e101-carrying-jobs.md, notebooks/E101_carrying_jobs.ipynb · E101 card, claims table, timeline · Amendment 1 (judge shown task windows, task-specific W0 controls) failed its power gate: 0 of 10 controls tied to their own task on each of three tasks, so the job reading is closed on the 67M; next Qwen3-1.7B (baseline-lead)
- 2026-11-05 · reports/plans/2026-11-05_plan_e102-qwen-carrying.md, experiments/TRIED.md (E102) · experiments (new E102 card, ongoing), timeline · E102 coded, not run: on Qwen3-1.7B, the SwiGLU neurons that carry the template and TOFU fine-tunes (E98's ranking), read for whether they were already engaged at the answers and already moved by the system turn, against a random change of the template's output size; control (i) first; two $0 debug jobs; the agent stage only if control (i) passes.
- 2026-11-05 mechanism-detection-lead: E100 Amendment 2 (jobs 52989, 53015, $0): site-relative bar not adopted (gap 0.19 under 0.22), fixed bars stand; emoticon keeps its gain with E62's copying heads off (1.01-1.02, control copy loses 89%) yet reads fed (0.30-0.37), so 'fed' is not only copy maps; class map and twin mostly between; sharpen pair stopped by its filed rule (17 strong keys under 24): column handed to definition-lead as the check worded it. E100's amendments are spent. Update the E100 card.
