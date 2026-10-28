# E38 Amendment 6: independent check of the joint and sign-split restores (review)

*2026-10-28, independent agent, read-only; no model loaded. Verdicts recomputed with `jq` from
`experiments/recruitment_test/results/amend6.json`, against `amend5.json` and `amend3.json`. First-order sums for sets
the run did not restore come from `results/large/amend5_scores.pt` and
`experiments/transport_test/results/large/alive.pt`, in one CPU job through `~/commute-agents/fleet/cpu-run`. Files
read: the plan `reports/plans/2026-10-28_plan_e38-amend6.md`, `predictions_amend6.json`, `run_amend6.log`,
`amend6.py`, the "E38 Amendment 6" section of `src/commute/recruitment.py` and the restore and scoring code it
reuses, and the Amendment 5 check.*

I recomputed every filed verdict of the joint restore and read the code that builds the sets and restores them.
Every verdict agrees with the run's own, and the code does what the filed definitions say. Restoring the switched-on
set together with the direction-matched set takes back nearly the sum of the two apart, so the filed tree reads
"separate" on both scored tasks, against the filed expectation of "partial". For `GOALS.md` Q5 this means the
switched-on set's share is not a duplicate of the direction-matched set's. It does not show that the switched-on set
holds anything a ranking by pull on the target would miss. Restores of disjoint sets add up nearly linearly here,
and the direction-matched set was barred from the switched-on set's members by construction.

## 1. Verdicts

Symbols: ρ(X) is the share of the gain taken back when the trigger outputs of set X are put back to their
original-model values, on held-out windows. S is the set Γ switched on. D is, per matrix, the same number of
subcomponents outside S with the largest first-order take-back e_c on the training windows. S⁺ and S⁻ are S's
members with training e_c above zero and at or below zero. The overlap reading is a = ρ(S ∪ D) − max(ρ(S), ρ(D)).
ê(X) is the held-out e_c summed over X, over the gain.

**Controls.** R0 to R4 pass on all nine states, with the same values as in Amendments 3 to 5. R7 passes on all six
task-seed states: ρ(S) and ρ(D), restored from the saved members, equal `amend5.json`'s with difference exactly 0.

**P14.** Every gain is 10.3 to 13.7 nats, so every state is scorable on the gain.

| state | ρ(S) | ρ(D) | ρ(S ∪ D) | a | separate bar 0.5 × min | seed class | run's |
|---|---|---|---|---|---|---|---|
| emoticon s0 | 0.239 | 0.352 | 0.487 | 0.135 | 0.120 | separate | separate |
| emoticon s1 | 0.220 | 0.397 | 0.539 | 0.142 | 0.110 | separate | separate |
| ` which` s0 | 0.179 | 0.193 | 0.335 | 0.142 | 0.090 | separate | separate |
| ` which` s1 | 0.202 | 0.180 | 0.343 | 0.141 | 0.090 | separate | separate |
| ` we` s0 | 0.037 | 0.114 | 0.146 | 0.032 | — | unscored | unscored |
| ` we` s1 | 0.049 | 0.116 | 0.151 | 0.035 | — | unscored | unscored |

Task classes: emoticon "separate", ` which` "separate", ` we` "unscored" (ρ(S) below 0.10 on both seeds, as the file
expected). Reading on ` which`: **"separate"**, with emoticon "separate" beside it, as the run wrote. The filed
expectation, "partial" on ` which` with ρ(S ∪ D) about 0.3, is missed: ρ(S ∪ D) is 0.335 and 0.343, and a clears the
separate bar by 0.05. That miss is a reversal of the filed direction, low confidence, and should be named as one.

Margins differ by task. On ` which` a sits 0.05 above its bar on both seeds, at the filed noise floor. On emoticon it
sits 0.016 and 0.032 above, inside the noise floor of 0.05. The filed rule reads emoticon "separate"; a claim should
rest on ` which`.

**The "reported" expectations.**

| expectation | recomputed | met | agrees with run |
|---|---|---|---|
| ρ(S⁺) ≥ ρ(D) − 0.05 on emoticon | 0.350 against 0.302; 0.360 against 0.347 | yes, both | yes |
| ρ(S⁻) ≤ −0.05 on emoticon | −0.071, −0.130 | yes, both | yes |
| ρ(S⁺), ρ(S⁻) on ` which`, no expectation | 0.192 and −0.014; 0.200 and −0.010 | — | reported as filed |

Sizes: S⁺ holds 127 and 129 of S's 210 and 213 members on emoticon, 39 and 46 of 56 and 62 on ` which`. On emoticon
S⁺ ties D on seed 0 and trails it by 0.04 on seed 1, with about 60% of D's count. First order overstates S⁺: ê(S⁺)
is 0.42 and 0.46 against ρ 0.35 and 0.36. On ` which` S⁻ does almost nothing, so S's share there is S⁺'s.

## 2. Code against the filed definitions

Read in `amend6.py` and `recruitment.py` lines 726-767. Neither changed between the code commit 3fd96aa and the
results commit 0bb4428. `predictions_amend6.json` has one commit, 3545487, before the code.

- **Sets read from saved members.** `saved_sets` reads S and D from `amend5.json`'s members per state, asserts the
  saved S equals this job's `run.S`, and asserts D has the same matrices and no member in common with S. R7
  reproducing Amendment 5's ρ to exactly 0 confirms the sets and the restore path are the same.
- **Union.** Per matrix, S's and D's index tensors concatenated and sorted. Disjointness is asserted, so there are no
  duplicates, and the restore hook applies each member's correction once.
- **S⁺ and S⁻.** `e_train[k][v] > 0` and `<= 0` on S's indices. `e_train` is Amendment 5's per-matrix (C,) training
  e_c, indexed by subcomponent, as `take_back` writes it. The two parts partition S, matching the filed definition,
  ties at zero going to S⁻.
- **Restore.** `amend3.restored` through `restored_at` and `restore_hooks`, unchanged since Amendment 3. A restored
  subcomponent's trigger output is set to W0's whatever happens upstream, so the union is one pass, not two
  corrections stacked.
- **ê.** Held-out e_c summed in float64, over the gain; it reproduces the run's ê(S⁺) and ê(S⁻) and Amendment 5's
  ê(S) and ê(D).
- **P14.** `overlap`, `p14_seed_class`, `p14_task_class` and `p14_reading` implement the filed rules, including the
  order "unscored, then separate, then redundant", and the reading taken on ` which` with emoticon beside it.

Nothing found makes a share, a set or a verdict wrong.

## 3. Alternatives to "S carries a part of the gain the top target-pointing subcomponents miss"

That sentence is the plan's card reading for "separate". The alternatives below test it.

**Ruled out by the data.**

- *S and D are two routes to the same part of the gain.* The union takes back 82% to 90% of the sum of the two apart
  on the scored tasks, far from the 50% that full redundancy would give.
- *The joint restore saturates, so the shares cannot add.* The plan expected this from ρ(all alive), 0.38 to 0.49.
  On emoticon ρ(S ∪ D) is 0.49 and 0.54, above ρ(all alive) of 0.38 and 0.37: a subset takes back more than every
  alive subcomponent together. So ρ(all alive) is not a ceiling. It is low because the alive set holds anti-target
  movers that cancel the rest. Amendment 5 already showed this to first order, with ê(all alive) at −0.42 and −0.41 on
  emoticon. The premise behind the "partial" expectation was wrong.
- *The emoticon gap between S and D comes from how strongly D's pieces point at the target.* S⁺, about 60% of S,
  takes back as much as D on seed 0 and 0.04 less on seed 1. S⁻ takes back −0.07 and −0.13. The Amendment 5 check's
  first-order reading holds when restored: the gap is mostly S's members that push against the target.

**Weakened but open.**

- *Restores of these sets interact strongly.* Splitting S into S⁺ and S⁻ is a second disjoint pair. Their restores
  add to within 0.04 of ρ(S) on every scored state: 0.279 and 0.230 against 0.239 and 0.220 on emoticon, 0.178 and
  0.190 against 0.179 and 0.202 on ` which`. Interactions exist, since first order overstates emoticon's union by
  about 1.5 times, but at the scale of these sets restores add nearly linearly.

**Surviving.**

- *Any two disjoint target-pointing sets would read "separate", so the reading says nothing special about S.* This is
  the strongest alternative. When restores add nearly linearly, a ≈ min(ρ(S), ρ(D)) less the shortfall from
  additivity. For two sets of equal share, "separate" needs only that the union reach 75% of the sum, and both
  disjoint pairs measured here reach 82% or more. "Redundant" can only be read when restoring one set makes the other
  stop mattering. Nothing in this run shows that S is special in that test. Amendment 3's random sets do not help
  either. They take back −0.01 to 0.00 on average, at most 0.029, which is what an additive restore of members with
  mixed signs gives. They show S's share is not generic. They say nothing about whether a second target-pointing set
  adds.
- *A ranking by pull on the target, allowed to take S's members, carries everything S ∪ D carries.* D was barred from
  S by construction. On the scored tasks, 27% to 37% of the per-matrix top |S| by training e_c are members of S, and 67% to 74% of
  the top 2|S| are in S ∪ D. So "the top target-pointing subcomponents miss S's part" holds only for the top
  subcomponents outside S. First-order sums, computed in this check and not filed: the unrestricted top 2|S| per
  matrix reach 0.43 and 0.44 on ` which` against ê(S ∪ D) of 0.36 and 0.36. On ` which` first order tracks restores
  to within about 10%, so this ranking would likely take back more than S ∪ D. On emoticon the same sums are 0.99 and
  1.13 against 0.73 and 0.80, but first order overstates there, so the restore is not predicted.

**Cheapest test that settles both.** One `debug` QoS job on the saved states, about a minute of L40, $0, with the
filed rule unchanged. Restore D₂ (the next |S| per matrix by training e_c outside S ∪ D) and D ∪ D₂; and T₂ (the top
2|S| per matrix by training e_c, S allowed). If a for (D, D₂) also reads "separate", the reading is generic. If ρ(T₂)
≥ ρ(S ∪ D), a target-aware ranking at the same count carries all that S ∪ D does. One caveat: on ` which` ê(D₂) is
0.09, just under the 0.10 floor. A clean (D, D₂) read there would take D₂ at the count that matches ê(S) instead; on
emoticon ê(D₂) is 0.21 and 0.25, near ρ(S).

**Context for any claim.** S, chosen without the target, holds a third of the top-direction subcomponents at its
count, about ten times the chance share (Amendment 5 check). On ` which` its share is almost all from those
target-pointing members. Amendment 6 adds that this share is not duplicated by D. It does not add a reason to prefer
S over a target-aware ranking.

## 4. What a claim in the research log may state

The tree reads "separate", so by the plan the card takes the "separate" reading. Its checked sentence, "S carries a
part of the gain the top target-pointing subcomponents miss", overstates. D could not contain S's members, and the
additivity seen here would make any disjoint pair with these shares read "separate".

May state: *On the 67M model after a 10-step fine-tune, we put back the trigger outputs of the switched-on set and
the direction-matched set together. Together they take back about a third of the gain on ` which` and about half on
emoticon, close to the sum of the two apart. So the two sets carry different parts of the gain, not one part twice.
Put-back shares of disjoint sets add almost linearly here, so this is what any two non-overlapping target-pointing
sets would show. On emoticon, the switched-on members that point at the target take back as much as the
direction-matched set, with about 60% of its count; the rest push against the target.*

May not state: that S holds a part of the gain a target-aware ranking misses, since D was barred from S and the
unrestricted top ranking contains a third of S; that "separate" shows anything particular to Γ, since no other
disjoint pair was scored with this rule; that the filed expectation was met, since it was "partial"; that restores
saturate, since the union exceeds ρ(all alive) on emoticon; that emoticon's "separate" is clear, since its margin
sits inside the noise floor; that S⁺ is a Γ-only set, since it is cut by e_c, which uses the target; or anything
about ` we` beyond "unscored", where S takes back about a twentieth.

On the plan's options: the result does not support "(b) upstream or (c) Qwen" on the ground that Γ's set is not
interchangeable with a target-aware ranking. That ground was not tested.

## Not run

- Any GPU computation by this check. D₂, D ∪ D₂, the unrestricted top 2|S| and the unrestricted top |S| were not
  restored; their first-order sums above were computed in this check and were not filed.
- S⁺ ∪ D, D's own sign split, and S ∪ D ∪ C_k.
- A third seed, a second-order or greedy ranking, other positions, other tasks, the larger model.
