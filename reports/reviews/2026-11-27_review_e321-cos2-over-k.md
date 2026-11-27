# Review: E321 (P1.65 Run 1), the graded cos² clause read over K

redteam-lead, 2026-11-27. Scope: `experiments/p165_cos2_k/{run.py,score.py}`, `results/{predictions.json,stage0_trained_s0.json,summary.json,run_score.log}` (commit 805debf7), plan `reports/plans/2026-11-26_plan_p1-65-readthrough-fills.md` Run 1, paper `origin/v2:paper/main.tex:145,432`.

The paper says that with one variable per neuron, a graded cos² wording reads every trained fine-tune as reused at 0.999 or more. That number came from a different set of variables than the one the definition uses, so this run re-read it over the definition's set. It still holds: the lowest of the 20 fine-tunes reads 0.99936, and the paper sentence can stand with its set label changed.

## Verdict: PASS, with one NIT and one caveat for the prose

- **Stage 0 passes as filed.** Do-nothing: 0 changed, dd max 0, E_b over K 1. Identity swap: 0 flips in 788,224 cases, largest logit deviation 5.05e-5 (filed: near E277's 4.6e-5).
- **Code.** `score.py`'s variable order (heads, neurons, remainder, per layer) matches `needed_neurons.variables`, which asserts that order. Mapping by part name left 0 parts missing on all 20 items. No remainder is in K, as expected at the width band.
- **Numbers, from summary.json.** The value over K ranges from 0.99936 (case_twin_s0) to 0.99994 (copy24_s0). The old set gives a minimum of 0.99938. All 20 values lie within the filed bound (the item's lowest changed-part value up to 1). Per item, the K value and the old set's value differ by at most 1.1e-4 (repeat_twin_s1: 0.99967 against 0.99978).
- **NIT (relay wording).** definition-lead's message names repeat_twin_s1 (0.999668) as the per-item low. The lowest item is case_twin_s0 at 0.999359; repeat_twin_s1 is the item that moves most from the old set. The line to writeup-lead should say case_twin_s0.
- **Caveat for the prose (not a MUST).** K holds 29-99% of the 12,316 variables (3,582 to 12,193), so the K value is close to the mean over all changed parts (0.99940-0.99995). The reading barely depends on which set it is taken over, because almost every part's cos² sits near 1. Individual parts go as low as 0.88. It is an unweighted mean (filed "blind to" line). This supports the sentence's point (the wording cannot tell fine-tunes apart). The sentence should not suggest that K selects anything.

## Paper edit this clears (main.tex:432, and :145 unchanged)
"scores each variable of $K^*$" becomes "of $K$", and `\kst{0.999 or more}` becomes a plain K value (min 0.99936 over the 20 E196 pair fine-tunes). Line 145 states no number and needs no change. This is writeup-lead's edit to make when the user allows paper edits.

## Not run
- No independent recompute from the raw npz: `results/large/items_*.npz` are not on disk in either clone. This is a transcription and code check only. The claim is not in the answer paragraph, so no independent check is owed.
- Run 2 (E322, rescale-only control seeds) is still on its pod. It gets a separate review.
