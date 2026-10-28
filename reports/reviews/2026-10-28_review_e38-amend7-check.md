# E38 Amendment 7: independent check of the next-ranked and twice-count restores (review)

*2026-10-28, independent agent, read-only; no model loaded. Verdicts recomputed with `jq` from
`experiments/recruitment_test/results/amend7.json`, against `amend5.json` and `amend6.json`. Set overlaps come from
the saved members with `jq`. First-order sums of the parts of T₂, and of a global top ranking the run did not
restore, come from `results/large/amend5_scores.pt` and `experiments/transport_test/results/large/alive.pt`, in CPU
jobs through `~/commute-agents/fleet/cpu-run`. Files read: the plan `reports/plans/2026-10-28_plan_e38-amend7.md`,
`predictions_amend7.json`, `run_amend7.log`, `amend7.py`, the "E38 Amendment 7" section of `src/commute/recruitment.py`
and the code it reuses (`top_per_matrix`, `amend6.saved_sets`, `amend6.shares`), and the Amendment 6 check.*

I recomputed every filed verdict of the two settling tests and read the code that builds the new sets. Every verdict
agrees with the run's own, and the code does what the filed definitions say. The top target-pointing set of twice the
switched-on set's size, free to include its members, takes back at least as much as the switched-on and
direction-matched sets together on every state, clearly more on emoticon and the same within noise on ` which`. The
test of whether any two target-pointing sets add up the way those two did fell outside the filed tree, because
emoticon's two seeds land on either side of the bar. For `GOALS.md` Q5 this means nothing in the switched-on set has been shown to
carry a part of the gain that a ranking toward the target misses at the same count. That set is a target-free pick rich
in target-pointing subcomponents, and its share is what those members carry.

## 1. Verdicts

Symbols: ρ(X) is the share of the gain taken back when the trigger outputs of set X are put back to their
original-model values, on held-out windows. S is the set Γ switched on, and n_k is its count in matrix k. D is, per
matrix, the n_k subcomponents outside S with the largest first-order take-back e_c on the training windows. D₂ is the
next n_k by the same ranking, outside S ∪ D. T₂ is, per matrix, the 2n_k subcomponents with the largest training e_c,
S's members allowed. a₂ = ρ(D ∪ D₂) − max(ρ(D), ρ(D₂)). ê(X) is the held-out e_c summed over X, over the gain.

**Controls.** R0 to R4 pass on all nine states, with the values of Amendments 3 to 6. R8 passes on all six task-seed
states: ρ(S), ρ(D) and ρ(S ∪ D), restored from the saved members, equal `amend6.json`'s with difference exactly 0.
No set is short in any matrix on any state.

**P15** (is Amendment 6's "separate" true of D and D₂?).

| state | ρ(D) | ρ(D₂) | ρ(D ∪ D₂) | a₂ | separate bar 0.5 × min | seed class | run's |
|---|---|---|---|---|---|---|---|
| emoticon s0 | 0.352 | 0.198 | 0.453 | 0.101 | 0.099 | separate | separate |
| emoticon s1 | 0.397 | 0.225 | 0.509 | 0.112 | 0.113 | partial | partial |
| ` which` s0 | 0.193 | 0.091 | 0.241 | 0.049 | — | unscored | unscored |
| ` which` s1 | 0.180 | 0.084 | 0.232 | 0.052 | — | unscored | unscored |
| ` we` s0 | 0.114 | 0.072 | 0.165 | 0.051 | — | unscored | unscored |
| ` we` s1 | 0.116 | 0.072 | 0.174 | 0.057 | — | unscored | unscored |

Task classes: emoticon "seeds disagree"; ` which` and ` we` "unscored", since ρ(D₂) is below the 0.10 floor on every
seed, as the plan foresaw. One task is scored and it is not "separate", so the reading is **"outside the tree"**, as
the run wrote. The filed expectation, "generic", is missed. It is not a reversal, since nothing reads "not generic".

The split on emoticon is a threshold effect. Seed 0 clears the bar by 0.002 and seed 1 misses it by 0.001. The
union-to-sum ratio, which the rule turns into a class, is 0.82 on both seeds.

**P16** (does T₂ take back as much as S ∪ D?).

| state | ρ(T₂) | ρ(S ∪ D) | difference | ê(T₂) | ê(S ∪ D) | seed class | run's |
|---|---|---|---|---|---|---|---|
| emoticon s0 | 0.622 | 0.487 | +0.134 | 0.991 | 0.729 | covers | covers |
| emoticon s1 | 0.696 | 0.539 | +0.157 | 1.129 | 0.795 | covers | covers |
| ` which` s0 | 0.362 | 0.335 | +0.027 | 0.433 | 0.358 | covers | covers |
| ` which` s1 | 0.358 | 0.343 | +0.015 | 0.436 | 0.364 | covers | covers |
| ` we` s0 | 0.188 | 0.146 | +0.042 | 0.208 | 0.139 | covers | covers |
| ` we` s1 | 0.198 | 0.151 | +0.046 | 0.221 | 0.164 | covers | covers |

Task classes: "covers" on all three. Reading on emoticon and ` which`: **"covers"**, as filed and as the run wrote.
On ` which` both margins are below the noise floor of 0.05 (section 3).

**The "reported" expectations.**

| expectation | recomputed | met | agrees with run |
|---|---|---|---|
| ρ(D₂) below ρ(D) on every state | yes on all six | yes | yes |
| \|T₂ ∩ S\| / \|S\| about a third | 0.357, 0.338 emoticon; 0.411, 0.484 ` which`; 0.217, 0.238 ` we` | on emoticon; above on ` which`, below on ` we` | yes |
| ê(T₂), no expectation | 0.99, 1.13; 0.43, 0.44; 0.21, 0.22 | — | yes |

**Together.** P15 is "outside the tree" and P16 is "covers". In the plan's decision tree that is the branch
"anything else: card with the numbers; the user picks". It is not the branch "P15 generic and P16 covers" that leans
to closing the 67M restore line. The card may lean there on the evidence, but it should not say the tree reached it.

## 2. Code against the filed definitions

Read in `amend7.py` and `recruitment.py` lines 768-814. `predictions_amend7.json` has one commit, a1945e0, before the
code commit 7dd5319. Neither the code nor the predictions changed in the results commit 4a56ace.

- **S and D.** Read through Amendment 6's `saved_sets`, with its asserts that the saved S is this job's S and that D
  is disjoint from S. `ranked_sets` also ranks D again from the saved training e_c and asserts it equals the saved D,
  so the ranking used for D₂ and T₂ is Amendment 5's.
- **D₂.** `top_per_matrix(e, alive, S ∪ D, n_k)`. D is the top n_k of the pool without S, so the top n_k of the pool
  without S ∪ D is ranks n_k + 1 to 2n_k of D's ranking, as filed. Disjointness from S ∪ D is asserted.
- **T₂.** `top_per_matrix(e, alive, {}, 2n_k)`, nothing excluded, as filed. Matrices with n_k = 0 get empty sets.
- **Ties.** `top_per_matrix` sorts by (−score, index), as filed. Ties occur on ` which`. In `h.0.attn.q_proj`, where
  n_k = 1, the pool's training e_c is exactly 0, as for many layer-0 subcomponents. D takes index 5, D₂ index 7 and T₂
  both, by index. These members take back nothing to first order. The Amendment 5 check's line "D's members all have
  positive training e_c" is off by this one tied member per ` which` state, which changes no number.
- **Short flags.** A matrix is short when its pool is smaller than the count. No matrix is short for D₂ or T₂.
- **Unions and restores.** D ∪ D₂ is the two index tensors concatenated and sorted, disjoint by assert. Every set is
  restored in one pass through `amend6.shares` and `amend3.restored`, unchanged since Amendment 3. R8's exact zero
  confirms the path.
- **ê.** Held-out e_c summed over the set, over the gain, as Amendments 5 and 6.
- **P15 and P16.** `p15_seed_class` passes a short D₂ in as unscored, then applies P14's rule to (D, D₂, a₂),
  floors first, "separate" before "redundant". `p15_reading` counts a "seeds disagree" task as scored, which gives
  "outside the tree", matching the filed "else". `p16_seed_class` reads "covers" on a tie and "short" only below
  ρ(S ∪ D) − 0.05. `p16_reading` reads emoticon and ` which` only.
- **Windows.** S comes from Γ's mean gate shift on the training windows. D, D₂ and T₂ come from training e_c. All are
  scored on the held-out windows, so S and T₂ had the same windows to choose from.

**A structural fact the plan does not state.** D is always inside T₂. Matrix k holds only n_k members of S, so at
least n_k of its top 2n_k lie outside S, and those are D. The rest of T₂ is the n_k members of S ∪ D₂ with the
largest training e_c. The saved members confirm this on all six states: T₂ holds all of D, and nothing outside S ∪ D ∪
D₂. So P16 compares S with the better-ranked half of S ∪ D₂, both added to D.

| state | \|S\| | T₂ ∩ S | T₂ ∩ D₂ | S members T₂ drops | of them, training e_c ≤ 0 | ê of the dropped S members | ê of the D₂ members taken |
|---|---|---|---|---|---|---|---|
| emoticon s0 | 210 | 75 | 135 | 135 | 83 | −0.116 | 0.147 |
| emoticon s1 | 213 | 72 | 141 | 141 | 84 | −0.155 | 0.179 |
| ` which` s0 | 56 | 23 | 33 | 33 | 17 | −0.010 | 0.064 |
| ` which` s1 | 62 | 30 | 32 | 32 | 16 | −0.013 | 0.059 |

T₂ drops every member of S with non-positive training e_c on every state. Nothing found makes a share, a set or a
verdict wrong.

## 3. Alternatives to the run's readings

The readings tested: P15's "Amendment 6's separate is generic"; P16's "a target-aware ranking at twice S's count
takes back as much as S ∪ D"; and the card reading for both together, "S's share is what target-pointing
subcomponents carry; Γ finds a set enriched for them without the target, and no more".

**Ruled out by the data.**

- *S ∪ D carries something the best first-order target-aware set of its size misses.* This is the plan's "short"
  branch. ρ(T₂) is at or above ρ(S ∪ D) on all six states: by 0.13 to 0.16 on emoticon, 0.04 to 0.05 on ` we`, and
  0.015 to 0.027 on ` which`.
- *Interactions make the first-order best set lose to S ∪ D.* The plan named this as T₂'s blind spot. It did not
  happen. Restores fall short of first order more for T₂ than for S ∪ D, but T₂ still does not lose.
- *T₂ wins by fitting the training windows.* T₂ is chosen on training windows and scored on held-out ones, so fitting
  the training windows could only lower its score. S is also chosen on training windows. Window access is not what
  separates them.

**Weakened but open.**

- *The additivity of S with D is particular to S.* On emoticon seed 0 the two pairs add identically, the union at 82%
  of the sum for both. On seed 1, (S, D) reaches 87% against 82% for (D, D₂). On ` which`, unscored for P15 and not
  filed, (S, D) reaches 90% on both seeds against 85% and 88%. So S adds to D a little more cleanly than the next
  target-pointing set does, by 0 to 5 points of the ratio. D and D₂ are adjacent in one ranking and sit in the same
  matrices, so some extra overlap between them is expected. Nothing here makes S's additivity special.
- *Amendment 6's "separate" is generic.* The filed test could not settle this. It is outside the tree because the
  emoticon seeds straddle the bar by 0.002 and 0.001, and D₂ is under the 0.10 floor on ` which`. The union-to-sum
  ratios above say the two pairs add almost equally. The plan's question is answered in substance, but not by the
  filed rule.

**Surviving.**

- *"Covers" on ` which` is equality, not an advantage.* The margins are 0.027 and 0.015. Both sit below the filed
  noise floor of 0.05, below the largest random-set share of 0.029, and near the 0.023 spread between S's seeds. The
  rule has no margin on the "covers" side: any non-negative difference reads "covers", and a difference between −0.05
  and 0 would read "unclear", not "short". The comparison is paired, on the same state and windows, with D and part of
  S shared. Its noise is probably below 0.05, but it was not measured. Both seeds have the same sign. So "covers" on
  ` which` survives as filed, as "T₂ takes back as much as S ∪ D". "T₂ takes back more" holds only on emoticon and
  ` we`, the latter not read.
- *T₂'s advantage is its use of the target.* By the structural fact above, T₂ is S ∪ D with S's lower-ranked members
  swapped for D₂'s best. On emoticon the swap drops all of S's members that push against the target, which Amendment
  6 measured at −0.07 and −0.13 of the gain. So most of emoticon's gap is a target-aware filter removing those members.
  This is what P16 asked, not a flaw. It means P16 compares Γ's set plus a target ranking with the target ranking
  alone. It is not a comparison of Γ with a target-free alternative. Amendment 3's random sets, near 0, remain the
  only target-free comparison, and S beats them.
- *On ` which`, S's dropped members carry more under restore than first order credits.* This is the strongest
  surviving alternative that leaves Γ something. To first order the swap should gain 0.075 and 0.072 of the gain on
  ` which`; restores gain 0.027 and 0.015. ρ(S) is 1.08 and 1.10 times ê(S) there, while ρ(D) equals ê(D). ρ(T₂)
  is 0.84 and 0.82 of ê(T₂), against 0.94 for S ∪ D. Either S's 33 and 32 dropped members take back more together
  than their first-order sum of about −0.01, or D₂'s best take back less than theirs. This run cannot tell, because
  neither part was restored alone. Even if S's dropped members do carry more, T₂ still matches S ∪ D, so this would
  show an effect beyond first order, not a share a target ranking misses.
- *T₂ is not Γ-free.* Its per-matrix counts come from S, as D's did. A global top ranking by training e_c at the same
  total count, computed in this check and not filed or restored, has held-out first-order sums 0.04 to 0.07 above
  ê(T₂) on every state. To first order, then, Γ's matrix allocation helps nothing either. Restores were not run.

**What this leaves of Γ's role.** Γ, given no target, picks a set holding a third of the top target-pointing
subcomponents at its count, about ten times chance (Amendment 5 check). That set takes back about a fifth of the gain
on emoticon and ` which`, far above random sets. Its share is accounted for by its target-pointing members (Amendment
6: S⁺ alone matches or beats S). A target-aware first-order ranking takes back as much at the same count, and adds to
it cleanly at twice the count. What survives is Γ as a target-free detector, the use the monitoring motivation needs.
Restores on this model have not shown that it finds anything a target-aware ranking at the same count would miss. The
one open hint is the ` which` gap between restores and first order above.

## 4. What a claim in the research log may state

May state, for Amendments 5 to 7 together: *On the 67M model after a 10-step fine-tune, putting back the original
trigger outputs of the subcomponents Γ switched on takes back about a fifth of the gain on emoticon and ` which`, far
above random sets of the same size. A set ranked by first-order pull toward the target, outside Γ's set and matched
to its count per matrix, takes back as much on ` which` and more on emoticon. Restoring the two sets together takes
back close to the sum of the two apart, as two adjacent target-ranked sets also do. The top target-ranked set of
twice Γ's count, free to take Γ's members, takes back at least as much as both together. It takes back clearly more
on emoticon and the same within noise on ` which`. So on this model nothing in Γ's set has been shown to carry a part
of the gain that a target-aware ranking at the same count misses. Γ finds, without the target, a set rich in
target-pointing subcomponents, and its share is what those members carry.*

May not state:
- that Amendment 6's "separate" was shown to be generic by the filed rule, since P15 is outside the tree; the
  union-to-sum ratios may be cited as unfiled;
- that the tree reached the branch that leans to closing the 67M restore line, since it reached "the user picks";
- that the target-ranked set takes back more than Γ's set with the direction-matched set on ` which`, since the
  margin is under the noise floor;
- that Γ's members carry nothing beyond first order, since on ` which` restores realised only a third and a fifth of
  the swap's first-order advantage, and the parts were not restored apart;
- that T₂ or D is Γ-free, since both take their per-matrix counts from S;
- that the result bears on Γ as a monitor, on positions other than the trigger, on the upstream step or on the larger
  model;
- anything about ` we` beyond the reported numbers, since it is unscored for P15 and not read for P16.

## Not run

- Any GPU computation by this check. The global top ranking's first-order sums, the overlap table and the
  first-order sums of T₂'s parts were computed here and were not filed.
- Restores of S ∖ T₂, T₂ ∩ S and T₂ ∩ D₂ apart, which would place the ` which` first-order gap.
- A noise estimate for the paired difference ρ(T₂) − ρ(S ∪ D), for example over held-out window resamples.
- D₂ at the count that matches ê(S) on ` which`, which the Amendment 6 check suggested for a clean P15 read there.
- A global top ranking restored, S⁺ ∪ D restored, a target-free ranking other than random sets.
- A third seed, a second-order or greedy ranking, other positions, other tasks, the larger model.
