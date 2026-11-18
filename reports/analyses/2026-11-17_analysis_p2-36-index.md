# P2.36: the paper's part-2 figures and appendices, from saved results

Three pages were written from saved results only, with no new runs, for writeup-lead to fold into the paper. The
share separates every built pair whether or not the update sizes are matched, while the prior and steering do not
track it; the older carrying line's figure shows teacher training as the only method that keeps the behaviour and
stays faithful on both fine-tunes. Nothing here gives a reuse/new verdict: the definition is the user's.

| Item | Page | Figure |
|---|---|---|
| (a) Figure 3a, share gap per pair, unscaled and size-matched, twins apart | `reports/analyses/2026-11-17_analysis_part2-figures.md` | `experiments/method_table/figures/paper_fig3a_share.{png,pdf,csv}` |
| (b) Figure 4, share gap vs prior gap and vs steering gap, E199 pair marked | same page | `experiments/method_table/figures/paper_fig4_share_vs_prior_steering.{png,pdf,csv}` |
| (c) Appendix "Baseline implementations" | `reports/analyses/2026-11-17_analysis_part2-appendices.md` (Appendix A) | none |
| (d) Appendix table "Per-pair numbers and recipes" | same page (Appendix B) | none |
| (e) Figure A1, behaviour kept vs adversarial damage ratio | `reports/analyses/2026-11-17_analysis_part2-figA1.md` | `experiments/split_comparison/figures/paper_figA1_carrying.{png,pdf,csv}` |

Scripts: `experiments/method_table/paper_figs.py`, `experiments/split_comparison/paper_figA1.py`. Every decimal on
the three pages resolves to a results file through its `.numbers.tsv`.

## Points writeup-lead and compare-lead should see
- The scorecard's lowercase-vs-ciphers size cell is the summed squared norm, labelled as its root in `final.json`;
  no call flips, since both sides share the unit (appendices page, inconsistencies item 1).
- Steering and the top-neuron share keep, per pair, the wider of two forms (`plot_final.py`, `wider`): a selection on
  the compared quantity, which the paper should state (item 6).
- Two step-10 cipher states are read at steps 8-9; one repeat-label seed is read at step 100 against step 30 for the
  others (items 2, 3).
- Size matching moves one pair's steering gap a long way; compare-lead's P2.28 report already calls it unexplained.

## Not run
- Size-matched readings for the four twin pairs, the untrained share and the neuron version (never run).
- Teacher training on the LoRA; the rescaling split on the SGD fine-tune (figure A1).
