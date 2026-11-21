# Two questions about the reuse definition: the memorising twins, and a score that sits near 1

*2026-11-20, definition-lead. Pending decision, both questions change what the definition or the paper says, so only the
user answers them. `decisions.md` and `GOALS.md` are unchanged. Source: writeup-lead's note of 2026-11-20 (two
definition questions found while redrafting the paper's Methods).*

## The answer

Nothing needs to be re-run to answer the first question; the second is best answered after a free reanalysis of files
already saved. The reuse score R has not read the built labels at any level so far, and at VPD it did the opposite of
what we filed. I recommend: say the twins test how the pairs were built, not R (question 1); and read a
behaviour-weighted R from saved files as a tested reading, not adopted, before anyone decides to keep or change the grade
(question 2).

## What has been measured, and whether it came out as expected

R is the mean grade over the smallest set of variables that keeps the fine-tune's behaviour. Each variable's grade is one
minus how far the fine-tune's write of that variable turned away from the base model's, as a squared cosine.

- **Coarse levels, saved size (E221).** The pair gaps followed update size more than any other reading (Spearman 0.54,
  0.69, 0.45), under the filed bar of 0.7. Not as hoped.
- **Coarse levels, size matched (E232, 2026-11-20).** We shrank the larger fine-tune of each of nine pairs to its
  partner's size. At blocks only 5 of 9 pairs kept the sign of their reuse-minus-new gap, against the 8 we filed, and a
  rule that uses update size alone predicted the sign of every pair. The filed prediction failed, as the size-only rival
  said it would. Report `reports/analyses/2026-11-20_analysis_e232-size-matched-r.md`.
- **VPD subcomponents (E226, 2026-11-20).** The edit built as reuse by multiplying eight copy neurons' outputs by ten
  scored R 0.9755, below the thinly spread edit built as new at 0.9963 (band 0.0010). The filed rule, every reuse item
  above both new edits, failed. Removing the search cap moved R by under 0.002 and kept the order, so the cause is the
  grade: a large gain on a few directions reads as a changed mechanism, while a small change spread widely reads as
  kept. On the nine trained pairs R lay between 0.9970 and 0.9997, gaps at most 0.00056, ranking with minus the size gap
  (Spearman 0.87) and not with the held-model reading (-0.02). Report
  `reports/analyses/2026-11-20_analysis_e226-kstar-vpd.md`.
- **Premise check on P1.37** (`~/commute-data/premise/P1.37.md`): doubtful. At four featurizations (P1.35, P1.36, P1.37
  and the neuron level) R tracks update size; refining the featurization made R look more like a size reading, not less.

So far, none of these came out as expected, and the neuron-level R (about 0.99 on 20 fine-tunes, per compare-lead's
E217) and the top-neuron R of E229 (0.9880 to 0.9982) also sit near 1.

## Question 1: do the memorising twins test R?

The paper's Properties say a fact memorised through small changes reads as mostly reused under the definition. Yet the
results use the memorising twins as a control that a measure must read low on. Under the definition a twin may then
correctly read high, so "a measure fails the twins" is not well defined.

**Options.**

- **(a) Say the twins test how the pairs were built, not R (recommended).** The twins stay in the results as a check that
  the paired fine-tunes differ the way we built them to differ. R is then judged only on its own properties. It tells us
  nothing new about R, and it removes a claim the definition does not support. This is Fable's suggested fix.
- **(b) Keep the twins as a low-reading control and state why the definition puts them low.** This needs a sentence in
  the definition that says a memorised fact stored by small changes is new. That contradicts the Properties as written,
  so the definition's wording changes. It would make the twins a real test of R.
- **(c) Drop the twins from R's results.** No wording changes. We lose the check that the pairs were built as labelled.

**Recommendation: (a).** It makes no wording change and the results stay as they were.

## Question 2: R sits near 1 and its gaps follow update size

R is an unweighted mean of a per-variable grade over a circuit that is nearly the whole model. So it sits near 1 at
neurons, at the coarse levels and at VPD, and what gaps remain follow how far the weights moved. At VPD the circuit
holds 9,601 to 34,839 of 38,938 variables on the six known items, and even the do-nothing circuit holds about two
thirds. A reading that weighted each variable by its effect on the behaviour would count the few variables that matter
for the behaviour more than the many that barely do. That changes how the definition averages, so it is not ours to
adopt.

**What is saved.** The user asked on 2026-11-20 that P1.37 save per-variable data for exactly this reanalysis. Checked in
`experiments/kstar_vpd/results/large/items_*.npz` (one per saved fine-tune, 68 pair states and 6 known items) and the
column notes inside them:

- **Exists:** the exact grade per variable `s` (row 0 at every position, row 1 at the read positions); exact circuit
  membership (`main_in_K_polished`, `main_in_K_prefix`); the exact changed-flag; and the split's norm shares per item
  (`split_shares`, in `results/items/<item>.json`).
- **Exists, approximate:** a signed ordering score per variable, `main_attribution`, 3 rows (the gradient of the
  divergence at the read position with respect to the variable's coefficient, at one half and at later search
  prefixes). The search orders by its absolute value. It is a first-order weight, not an exact one.
- **Does not exist:** an exact single-variable ablation drop. The search never ablates one variable alone in the full
  fine-tune, and the user asked that no new pass be added for it.

A behaviour-weighted R from the approximate weights is a $0 reading from files on disk. Its limit: first-order weights
may mis-rank variables that matter jointly, which the search diagnosis found for VPD's inactive subcomponents
(`reports/analyses/2026-11-20_analysis_e226-search-cap.md`).

**Options.**

- **(a) Keep R as worded and report saturation and size tracking as findings about this definition.** The paper says
  plainly that R near 1 and size-ordered gaps are what this definition gives. Cheapest, and an honest result; it leaves
  the paper with no measure that reads the built labels.
- **(b) Read a behaviour-weighted R as a tested reading, not adopted.** Weight each variable's grade by its approximate
  effect on the behaviour, on the six known items and the nine pairs, from saved files, at no compute cost. It tells
  us whether weighting separates built reuse from built new where the unweighted R did not, and whether the pair gaps
  still follow size. No wording changes. Its result gives the user numbers for (a) or (c).
- **(c) Change the grade**, for example to one that sees a change of size instead of a squared cosine. This is a
  wording change only the user makes, and it needs a new cell and runs. It would address the x10 reversal directly.

**Recommendation: (b) first**, as a reading with no adoption and no wording change, so you choose (a) or (c) with its
numbers in hand. If it fails to separate the known items, (a) is the likely honest endpoint.

## What I need from you

1. Twins: (a), (b) or (c)? I recommend (a).
2. Saturation: may I read the behaviour-weighted R from the saved files, as a tested reading and not adopted? I recommend
   yes, and no wording change meanwhile.

## Files

- E226 report: `reports/analyses/2026-11-20_analysis_e226-kstar-vpd.md`
- E232 report: `reports/analyses/2026-11-20_analysis_e232-size-matched-r.md`
- Premise check: `~/commute-data/premise/P1.37.md`
- Saved per-variable data: `experiments/kstar_vpd/results/large/items_*.npz`, `experiments/kstar_vpd/results/items/*.json`
- Source note: `~/commute-data/agents/definition-lead/notes-read/note-2026-11-20-writeup-twins-saturation.md`
