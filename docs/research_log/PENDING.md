# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
- 2026-10-30 baseline-lead · reports/plans/2026-10-30_plan_baseline-next-draft.md · Pending decisions (new card), E54 card (What next) · Pending decision: approve E56 (Fable's draft)? Neuron patching (E52's B2, an input-dependent cheap baseline) on E54's six saved states, then seeds 2-4 of the elicit/cipher pair, one $0 MATS debug experiment. Why: B2 tests the sharpest objection (the pair is steering-hard by construction), and more seeds tighten the ratio floor (at least ~6, unbounded above, on two seeds). Options: approve; run Jain et al. Def 1 readout first; wait for Qwen (not ready: no route split on Qwen until Q3; E45 anchor 1 on its fallback). Run line: none until approved.
