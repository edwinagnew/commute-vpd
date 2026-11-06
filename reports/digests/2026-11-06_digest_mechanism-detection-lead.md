# Mechanism detection on the 67M, 2026-11-06: finished, nothing approved left

*mechanism-detection-lead. Why finished: E110 and E114 are done and logged (log v266), and every next step is a new
experiment for the user to choose. $0 spent today (cluster debug only). No jobs, pods or subagents running.*

**What ran.**
- **E110**: answered from E100's saved reads, no run. The lowercase pair (1,1)+(1,2) is needed but unchanged, in its
  weights and in its output. The lowercase gain is carried by the MLPs of blocks 2 and 3. Addendum in
  `reports/analyses/2026-11-05_analysis_e97-locator.md`.
- **E114**: job 54238, about 12 min. The locator ran on E112's four case-label states, and every control passed. No part
  is located. The real fine-tune and the twin need the same parts: the MLPs of blocks 0-2 and head (1,4). That agrees
  with the definition's "not through the capability". Head (1,2) and block 3's MLP favour the real fine-tune but stay
  under the 0.55 bar. The sum of single reverts is lower on the real fine-tune (0.37, 0.48 against 0.72, 0.68). But the
  twin's update is three times larger, so that sign is weak. Report
  `reports/analyses/2026-11-06_analysis_e114-locator-case-label.md`; log card E114.

**Context.** The reuse definition was reset to Jain et al. today and frozen (`decisions.md` rows 164-165). My reads sit
beside it as evidence and are never a rule in it. Notes went to definition-lead and baseline-lead.

**Open for the user (not filed as asks; nothing waits on them).**
1. Read a twin saved at the real fine-tune's update size, to test whether the sum-of-reverts sign survives.
2. Read what head (1,2) attends to at the marker.

**Friction.** One lead-brief error: E114's writer was told to score held-out windows, where the twin has no gain by
construction. The writer caught it (friction 583).
