# Identity test: the components keep their identity, but the update is spread over thousands of them

**Takeaway.** Under the pull-back, 99% of alive subcomponents change by under 1% (relative change of u vᵀ), so every
one is trivially "still itself". The emoticon behaviour gained is carried by thousands of tiny changes: reverting the
10 most important removes 1.6% of it and the top 1 000 remove 36%. The paper's emoticon subcomponent `2359` is not
strengthened (scale ratio 1.000). Faithful and identity-preserving, yes; *explained by* a few components, no.

*2026-10-15, coding agent. Plan `reports/2026-10-15_plan_identity-test.md`; predictions
`experiments/identity_test/results/predictions.json` (filed before running); results `results/identity.json`. RunPod
4090, $0.35, about 9 min. Reproduction of round 2 checkpoints: |ΔP(o)| ≤ 1.2e-3, |Δ rel. norm| ≤ 7e-5.*

**Terms.** *Revert-one drop*: the fall in held-out log-odds of `o` when one subcomponent alone is set back to its W0
value. *Gain*: log-odds at the state minus at W0. *Selected*: the 20 largest revert-one drops. *Null group*: one
low-drop subcomponent from the same matrix as each selected one. *Keeps identity*: x·v profile correlation ≥ 0.9 with
its old self, the best match among all alive subcomponents of its matrix, and ablation-effect cosine ≥ split-half
floor − 0.1.

## Scores

- **P1 (identity kept): 20 / 20, passes as filed, but has no power.** The null group passes equally (20/20, r_self
  0.999). The selected subcomponents change by 0.22% (step 10) and 0.30% (step 30) in ‖u vᵀ‖, less than the null
  group (0.29%, 0.38%). With changes this small the metric cannot fail. No positive control was run (a subcomponent
  deliberately swapped or rotated, which the metric must flag), so under the 2026-10-15 rule in `.claude/CLAUDE.md`
  P1 should be read as **uninformative**, not as evidence of interpretability.
- **P2 (update concentrated): null outcome, as predicted.** Top 10 reverted together remove 0.016 of the gain (pass
  needed ≥ 0.5; null < 0.2).

## Concentration (fraction of gain removed by reverting the top k by single drop)

| State | Gain | k = 10 | k = 100 | k = 1000 | all |
|---|---|---|---|---|---|
| pull-back step 10 | 15.0 | 0.016 | 0.087 | 0.37 | 1.000 |
| pull-back step 30 | 11.6 | 0.016 | 0.083 | 0.36 | 1.000 |
| lift, W step 2 (matched to pull-back step 10) | 16.6 | 0.045 | 0.212 | 0.54 | 0.726 |
| lift, W step 3 (matched to pull-back step 30) | 17.7 | 0.042 | 0.202 | 0.53 | 0.700 |
| lift, W step 10 | 13.5 | 0.036 | 0.170 | 0.49 | 0.573 |

- **Largest single drop**: 0.03 log-odds (pull-back) or 0.10 (lift), against a gain of 11–17.
- **The lift is about 2.5× more concentrated** than the pull-back at matched norm, though still diffuse. Its "all" is below 1 because
  the part of δW left in Δ is not reverted.
- **`2359` and its five siblings** are not among the top 10 in any state. Under the pull-back their drops are
  0.004–0.008 and their scale ratio is 1.000: the subcomponent the paper says implements emoticon → `o` is not where
  the pull-back put the update.

## What this does and does not say

- **Says:** the pull-back learns the behaviour by moving almost every alive subcomponent a tiny amount. That keeps the
  decomposition faithful (round 2) and every component's identity intact (here), but the new behaviour is not
  attributable to a few components, nor to the one the decomposition labels as "the emoticon component".
- **Does not say:** whether a concentrated solution exists inside the decomposition (round 1 found 10 edited
  subcomponents suffice, but 10 random did as well). It does not say whether the diffuseness comes from Adam, which rescales each
  coordinate and so spreads small steps everywhere, or from the task. Revert-one is a marginal measure; joint effects
  of groups other than the greedy top-k were not searched.

## Not run

- A positive control for the identity metric (swap or rotate one subcomponent; the metric must flag it).
- Pull-back with plain SGD, or with a sparsity penalty on which subcomponents may move, to see if the update
  concentrates (and whether into `2359`).
- The same attribution for the W fine-tune itself, by component share of δW, without the lift.
- Longer pull-back training (beyond 30 steps), where per-component changes would be large enough for identity to fail.
- Behaviours other than the emoticon task.
