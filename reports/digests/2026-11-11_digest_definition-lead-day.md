# definition-lead, 2026-11-11 (day): finished, waiting on the gate (11-13)

**Cells.** All 13 of definition-lead's cells are filled or closed: P1.1, P1.2, P1.3 (1/1 runs), P1.4 (1/3), P1.6 (1/1), P1.7 (1/2), P1.8,
P1.9 (1/2), P1.12 (1/1), P1.13 (1/1) filled; P1.5 (0/2), P1.10 (1/1), P1.11 (1/1) closed. L1.4 has no run budget left. Finished
because no open cell has work left, and the next step (adopt, revise or drop the wording) is the user's at the gate.

**What ran today.** The built control, four attempts and a diagnosis, all on cluster debug at $0. No built control has yet
read "new" cleanly. The clearest reuse case, a push on whether the last number is odd, read just under the 0.5 cut-off. The
diagnosis (P1.13, E156, job 58855) gave no call: the search reads 0.84 even on an edit built to be exactly reachable, wider
maps do not raise the control by the filed margin, and two sites in order put it just above 0.5. So the shortfall is in the
scale of the measure. Page: `reports/analyses/2026-11-11_analysis_h-diagnosis.md`.

**For the gate.** The gate page `reports/analyses/2026-11-10_analysis_gate-page-working-definition.md` now has the
built control as run and an open question, put without a recommendation: what value should the wording give an edit known
to be exactly linear, and how do the 0.5 and 0.8 cut-offs sit against it? Also yours: whether two sites count as one map.

**Asks open:** none. **Spend:** $0 (no pods). **Jobs:** none live. **Log:** two lines queued in `PENDING.md` for
compare-lead's clerk. **Friction:** 709 (python heredoc refused on dev), 710 (worktree guard refused a chained push).
