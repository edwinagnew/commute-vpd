# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.

- 2026-10-29 reuse-lead · `reports/analyses/2026-10-29_analysis_e55-stage-a.md` (new section "Amendment 4", opening and answer updated), `notebooks/E55_pcfg_anchors.ipynb` (Amendment 4 cells, answer updated), `experiments/TRIED.md` E55 row, figure `experiments/pcfg_anchors/figures/amend4_offlabel.png` (from `plot_amend4.py`) · E55 card (add Amendment 4 to Found; figure fig/e55_amend4_offlabel.png), pending-decision card decision-e55-truth-set (replace the either/or line "Either they count b there, or they strengthened the count of a (not yet checked)" with: checked on swapped rows where the two give different answers, the two all-spurious anchors answer through a strengthened count of a, narrowly and with a pull from b, not a count of b; the 80% anchor reads neither there), decision-e55-plan run line ("Stage A and Amendments 1-4 done"), timeline 2026-10-29 · E55 Amendment 4 ran (job 44746, $0): off-label swap-built rows; W1 1K and seed 1 read "reads a" (B_a 0.67, 0.70; P_a 0.505, 0.496 vs bar 0.483; B_b 0.001), M3 1K neither, N1 1K reads b (control valid); P1, P1s met, P2 missed; reading R-a (existing count made stronger) for the all-spurious anchors; bears on the Stage C truth set, does not decide it.
