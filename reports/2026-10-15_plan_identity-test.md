# Plan: the identity test. Are the subcomponents that carry the update still the same subcomponents?

*2026-10-15, coding agent. Test 1 of the 2026-10-15 check, confirmed by the user ("do 1"). Predictions:
`experiments/identity_test/results/predictions.json` (filed before any measurement). Code: `experiments/identity_test/run.py`,
`src/commute/identity.py`, CPU tests `tests/test_identity.py`. Runs on RunPod (one 4090; the user preferred speed).*

**Question.** Round 2's pull-back keeps the decomposition faithful. Does it also keep it *interpretable*: are the
subcomponents that now carry the emoticon update the same subcomponents as before, and is the update carried by a few?

**Selection.** Selection is functional, not by Frobenius change (`TRIED.md` row 1). For every alive subcomponent, revert it alone to its W0 value and record
the drop in held-out log-odds of `o`. The top 20 are "selected". Null group: one subcomponent with a below-median
drop from the same matrix as each selected one.

**Identity, gauge-free** (`theory/component_identity.md`, claims 1 and 3):
- read: correlation of x · v before and after over Pile rows 512–543 plus the held-out windows; within-matrix
  retrieval rank; top-100 Jaccard;
- write: the ablation effect on the next-token distribution at its top-64 positions, cosine before vs after, against
  a split-half floor at W0;
- Γ reads raw activations and is frozen, so its profile only drifts through upstream weights. It is reported, not
  scored.

**Scored (pull-back step 30).** P1: ≥ 18/20 selected keep identity (r ≥ 0.9, rank 1, write cosine ≥ floor − 0.1);
null ≤ 10. P2: reverting the top 10 removes ≥ 50% of the log-odds gain; predicted to fail (diffuse).
**Reported.** Pull-back step 10; the lift at matched norm (W steps 2, 3) and at step 10 (PGD 5.4); `2359`'s rank.

**What would change next.** If identity holds and the update is diffuse, the pull-back is faithful but spreads the update
over many subcomponents, so "inside works" does not mean "explained by a few". If identity fails under the
pull-back, faithfulness without identity is the finding. If the lift loses identity where the pull-back keeps it,
that is a candidate cause of its PGD fragility.
