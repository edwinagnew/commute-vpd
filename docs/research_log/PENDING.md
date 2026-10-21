# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
A flush applies every line to `index.html`, republishes, and moves the lines to the changelog row it writes, leaving
this list empty. Recipe: `README.md` → "Batched updates".

(empty: flushed 2026-10-21 into v24)
- 2026-10-21 · reports/2026-10-21_analysis_cancellation.md, notebooks/E17_cancellation.ipynb · E17 card (already edited in index.html: new headline, two new figures), Q6 status sentence · E17 write-up rewritten for readability, numbers unchanged: question and the two readings that motivated it first, half-revert reference values drawn (toy arrows; dashed curve h = ¼(1+1/κ)), control failure explained (weight-space cancellation that barely reaches the output; does not rescue cancellation), remainder marked as inferred with the follow-up being built. Figures: fig/e17_half_revert.png ← experiments/cancellation/figures/half_revert.png (now coherence vs half revert), fig/e17_adversary_revert.png ← experiments/cancellation/figures/adversary_revert.png (now share of the excess, stacked); coherence.png deleted, folded into half_revert.png.
