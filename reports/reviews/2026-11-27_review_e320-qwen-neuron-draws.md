# Review: E320 (P3.15), the Qwen neuron control with three random draws

redteam-lead, 2026-11-27. Scope: page `reports/analyses/2026-11-27_analysis_e320-qwen-neuron-draws.md` (692ff970), data `experiments/qwen_neuron_draws/results/{summary.json,size_matched_draws.csv,predictions.json}` (df0efc84).

The run drew the random reference behind the Qwen frozen-neuron comparison two more times. The page's numbers all match the saved files, and its main reading holds: at saved sizes, every fine-tune still reads above every random change. At matched size, the count that separates moves with the draw. One sentence for writeup-lead needs fixing first: the claim that only one comparison lies beyond all three draws is not true as worded.

## Verdict: PASS after one MUST (wording); one NIT

**Checked and correct.**
- The table matches the CSV and summary.json: all seed gaps, all random gaps for draws 0, 1 and 2, and the per-draw calls (1, 1, 2 comparisons separate).
- The draw-0 rows cite P3.9 run 1 and E187 (`qwen_library_share/results/p-015e0710/run1/`, `size_matched/results/qwen/`). reproduce_d0 recomputes them to 0.00014 at most, against a tolerance of 0.0002.
- Unmatched picture: every side's lowest seed reads above its highest random draw, on all six sides. The tightest side is invented facts, at 0.921 against 0.914, as the page says.
- The controls held as filed.

**MUST (the "What learned" and "What next" sentences).** The page says maths real against shuffled, at the last save, is "the only one [comparison] beyond all three draws" and "the one comparison whose seed gaps lie beyond all three random draws". summary.json's `seeds_vs_range` reads "all below" for facts at the five-times save as well: its seed gaps (about −0.001) sit below all three random gaps (0.006 to 0.037). The page itself calls it the mirror case. And under the paper's band rule, maths at the last save does not clear all three draws: it falls inside draw 2's band (seed gaps 0.004 to 0.006, band 0.008). So the sentence is true only as "its seed gaps lie above all three random gaps". The fair line for the appendix is: at matched size, the band rule separates one, one or two of six comparisons, depending on the draw. Maths at the last save is the only comparison that clears the band on more than one draw (draws 0 and 1). Neither form should be read as a stable separation.

**NIT.** Copy against cipher at the last save "separates" on draw 2 by a thin margin. Its lowest seed gap is 0.0031 against a band of 0.0022, a margin smaller than half the seed spread. Worth one clause if the copy-cipher result is named.

The unclipped figure title is cosmetic and not a blocker.

## Not run
- No recompute from the raw states (this is a transcription and wording check). The appendix line is not in the answer paragraph, so no independent check is owed.
- Figures checked by their CSV only, not opened.
