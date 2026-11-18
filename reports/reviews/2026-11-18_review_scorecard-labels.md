# Scorecard labels: two points from the P2.36 appendix (2026-11-18)

The P2.36 appendix writer (regating-lead, `reports/analyses/2026-11-17_analysis_part2-appendices.md`, "Where the code
and the existing labels disagree") found points where the part 2 scorecard's labels differ from its code. compare-lead
checked two of them; no number changed.

1. **Lowercase against ciphers, size column.** The cell holds `norm2` from `src/commute/update_spectrum.py`, which is
   the summed squared matrix norm (line 6: `norm2 ||dW||^2`), but `experiments/method_table/plot_final.py` labelled it
   "root of summed squared matrix norms". Fixed on 2026-11-18: the label in `plot_final.py` and the saved
   `experiments/method_table/results/final.json` now read "summed squared matrix norms (not rooted)". Both sides of the
   pair share the unit, so no within-pair call changes; its gap is not comparable in size with other rows' gaps. The
   2026-11-10 method-comparison report and the P2.11 report keep the old label as written; read them with this note.
2. **`plot_final.py:wider`.** Steering and the top-neuron share each have two forms, and the scorecard keeps per pair
   the one with the wider gap. This is a selection on the compared quantity. It favours those two baselines over the
   share, so it can only make the share look like it adds less, not more. Left as filed; to go in the part 2
   "not shown" list.

Not run: nothing re-read; no figure redrawn (the figure prints no unit for this cell).
