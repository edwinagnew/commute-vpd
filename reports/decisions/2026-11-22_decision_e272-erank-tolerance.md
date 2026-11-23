# P2.66 blocked, due 11-24: E272's stage 0 stopped on an effective-rank tolerance below rounding noise

**What has been measured so far.** E272 is the realistic convergence panel: nine ordinary fine-tunes of the 67M, with each reading of reuse compared with blind raters' sort of the tasks and with the other readings. Its stage 0 runs the controls alone, before any training. It re-reads one saved fine-tune from E211 (lowercase, seed 0, size 0.0175) and checks that three readings repeat their saved values:
- the re-gating share read 0.7693 against 0.7698 saved (tolerance 0.01): **pass**;
- steering read 0.75133 against 0.75141 saved (tolerance 0.001): **pass**;
- effective rank read 597.71444 against 597.71408 saved (tolerance 1e-6, absolute): **fail**. The difference is 3.6e-4, or about 6 parts in 10 million of the value.

The job stopped as its rules require (cluster job 63620, $0, about one minute of run time). The checks after this one never ran: the E187 copy-versus-cipher split, and whether the 67M learns SQuAD and CoNLL on held-out data. Nothing has been trained and nothing has been spent.

**Why it failed.** The 1e-6 bar was copied from E211's own filed tolerance. E211 compared readings made on one code path in one job. This job reads the saved bf16 state on a different card and through a different path, so the singular values differ in the seventh significant figure, which is float32 rounding. `.claude/CLAUDE.md` asks for agreement bars that are relative and taken from a measured floor between the two paths, never an absolute guess. This bar was neither, so it is a mis-specified control. No sign points to a code bug: the share and steering re-reads agree well within their bars.

**What the pending run measures that nothing yet has.** Whether readings of reuse on realistic fine-tunes follow intuition (hope 1) and agree with each other (hope 2). Stage 0 only guards the apparatus.

**The question.** May E272's effective-rank reproduction bar become relative, 1e-4 of the saved value (about 0.06 here, roughly 150 times the gap measured), with stage 0 rerun as "E272 rerun: erank bar relative"? The other bars stay as filed.

**Options.**
1. **(Recommended)** Make the bar relative at 1e-4 in a dated amendment that records both tolerances, and rerun stage 0. Training goes ahead only if every other control passes. This tells us whether the rest of the apparatus (the E187 split and SQuAD/CoNLL learnability) holds tonight. Cost: $0, about 10 minutes.
2. Set the bar to the measured gap times 3 (about 1.8e-6 relative). This is tighter, but it rests on one measured pair.
3. Drop effective rank from the reproduction check and keep it as a reading. Its reproduction would then go unchecked.
4. Wait for the user in the morning. The panel then starts about 10 hours later than it could, and the cell is due 11-24.

**Evidence.** `experiments/convergence_panel/results/stage0.json` (key `e211_reread`); log `cluster/logs/commute-convergence-panel-63620.out`; the filed bar is `experiments/convergence_panel/results/predictions.json` → `thresholds.reproduce_tol`. Plan: `reports/plans/2026-11-22_plan_convergence-panel.md`.
