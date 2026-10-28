# E38 Amendment 5: independent check of the direction-matched restore (review)

*2026-10-28, independent agent, read-only; no model loaded. Verdicts recomputed with `jq` from
`experiments/recruitment_test/results/amend5.json`; per-subcomponent breakdowns from `results/large/amend5_scores.pt`
and `experiments/transport_test/results/large/alive.pt` in one CPU job through `~/commute-agents/fleet/cpu-run`.
Files read: the plan `reports/plans/2026-10-28_plan_e38-amend5.md`, `predictions_amend5.json`, `run_amend5.log`,
`amend5.py`, the "E38 Amendment 5" section of `src/commute/recruitment.py` and the restore code it reuses, `amend3.py`'s
`restored` and `w0_pass`, `amend2.py`'s `lo_ref`, `amend4.json`, and the Amendment 4 check.*

I recomputed every filed verdict of the direction-matched restore and read the code that scores each subcomponent,
picks the comparison set and restores it. Every verdict agrees with the run's own, and the code does what the filed
definitions say. The set picked by each subcomponent's first-order pull on the target takes back more than the
switched-on set on emoticon and about the same on ` which`, so the switched-on set is never ahead. The filed tree
reads "outside the tree" only because ` which` is a tie. For `GOALS.md` Q5 this means Γ's set does not carry more of
the change than target-pointing subcomponents do. It does find many of them without being told the target.

## 1. Verdicts

Symbols: ρ(X) is the share of the gain taken back when the trigger outputs of set X are put back to their
original-model values, on held-out windows. S is the set Γ switched on. D is, per matrix, the same number of
subcomponents outside S with the largest first-order take-back e_c on the training windows. C_k is the same per-matrix
count chosen by unsigned output change.

**Controls.** R0 to R4 pass on all nine states, as in Amendments 3 and 4. R6 passes on all six task-seed states:

| state | R6 agree (need 8 of 10) | run's |
|---|---|---|
| emoticon s0, s1 | 10, 10 | ok, ok |
| ` which` s0, s1 | 9, 9 | ok, ok |
| ` we` s0, s1 | 10, 10 | ok, ok |

The one disagreement on ` which` is the same subcomponent on both seeds, `h.3.mlp.c_fc` 1839: e_c is −0.30 and −0.45
nats, but restoring it alone takes back about 0.03 nats instead of adding gain. R6 tests sign only; the largest single effects also differ in size from their
first-order estimate, from about 0.2 to 1.2 times.

**P13.** Every gain is 10.3 to 13.7 nats, so every state is scorable; D is short in no matrix.

| state | ρ(S) | ρ(D) | ρ(S) − ρ(D) | seed class | run's |
|---|---|---|---|---|---|
| emoticon s0 | 0.239 | 0.352 | −0.113 | not ahead | not ahead |
| emoticon s1 | 0.220 | 0.397 | −0.177 | not ahead | not ahead |
| ` which` s0 | 0.179 | 0.193 | −0.013 | not ahead | not ahead |
| ` which` s1 | 0.202 | 0.180 | +0.022 | unclear | unclear |
| ` we` s0 | 0.037 | 0.114 | −0.077 | not ahead | not ahead |
| ` we` s1 | 0.049 | 0.116 | −0.068 | not ahead | not ahead |

Task classes: emoticon "not ahead"; ` which` "seeds disagree"; ` we` "S too small" (ρ(S) below 0.10 on both seeds).
Two tasks are scored, but only one is "not ahead", short of the two that "direction suffices" needs; none is "S
ahead". Reading: **"outside the tree"**, as the run wrote. The filed expectation, "direction suffices", is missed on
` which` only.

The ` which` split is not a real disagreement between seeds. Both differences, −0.013 and +0.022, sit inside the
filed noise floor: S's own two seeds differ by up to 0.023, and random sets never exceeded 0.029. Read plainly,
` which` is a tie.

**The "reported" expectations** (emoticon and ` which` where the file says so):

| expectation | recomputed | met | agrees with run |
|---|---|---|---|
| ρ(S) equals Amendment 4's to 1e-4 | difference 0.0 on all six | yes | yes |
| ρ(C_k) below ρ(S) − 0.05 | emoticon −0.123, −0.146; ` which` 0.097, 0.100 | yes, all four | yes |
| top-direction share of S below 0.5 | 0.271, 0.277, 0.304, 0.371 | yes | yes |
| ê(S), ê(D) within a factor 2 of ρ | ratios 0.91 to 1.29 | yes, all eight | yes |
| ρ(D) above 0.10 | 0.352, 0.397, 0.193, 0.180 | yes | yes |
| ê(all alive) against Amendment 4's ρ(all alive) | emoticon −0.42, −0.41 against +0.38, +0.37; ` which` 0.13, 0.17 against 0.46, 0.49; ` we` 0.20, 0.31 against 0.41, 0.44 | no expectation filed | reported as filed |

**The unfiled row matters.** The plan named ρ(all alive) as the reference for whether first-order shares add up. On
emoticon the first-order sum over all alive subcomponents has the wrong sign. It also misses C_k: ê(C_k) is −0.61 against
ρ(C_k) −0.12 on emoticon, and −0.15 against +0.10 on ` which`. So the first-order score adds up for S and D, which hold
small target-pointing pieces, and fails for sets holding the large anti-target movers. The P13 comparison uses ρ, not
ê, so this does not touch the verdicts. It limits any wider statement that "first-order effects explain restores".

## 2. Code against the filed definitions

Read in `recruitment.py` lines 579-722 and `amend5.py`; neither changed between the run script's commit 5574529 and
the results commit 419845a.

- **The gradient g.** `trigger_grads` adds a zero leaf to every hooked matrix's output at the trigger and takes one
  backward per batch of the trigger log-odds summed over windows. The forward is unchanged, so this is the total
  derivative through the plain W1 pass, as filed. Windows do not interact, so each row is that window's own gradient.
  The leaf is in bf16 under the same autocast as `readout`. That costs precision, not sign; R6 and ê against ρ bound
  the effect. The log-odds is `identity.log_odds` at `half`, the same quantity `readout` averages as lo.
- **e_c.** `take_back` computes the mean over windows of ((x1 − x0)·v_c)(g·u_c) in fp32, from W1's and W0's
  `trigger_io` inputs on the same windows and batches. Restoring c alone adds ((x0 − x1)·v_c)u_c, the restore
  hook's correction for one subcomponent with upstream inputs unchanged, so e_c is minus its first-order change in lo.
  The sign matches the definition: positive takes back gain.
- **Window sets.** D, C_k and the top-direction share use training e_c and training Δ_c. ρ and ê use the held-out
  windows. The training and held-out e_c agree closely: rank correlation over all 9881 alive subcomponents is 0.94 to
  0.98.
- **D and C_k.** `top_per_matrix` takes, per matrix, the |S ∩ k| alive subcomponents not in S with the largest score,
  ties by index, and flags a short pool. D's members all have positive training e_c. C_k uses Amendment 4's Δ_c.
- **Top-direction share.** The same function with nothing excluded, then the members of S among each matrix's top.
- **R6.** The ten alive subcomponents with largest |training e_c|, each restored alone on the training windows against
  lo_train(W1), and the filed agreement count.
- **Restores.** Unchanged from Amendment 3; ρ(S) reproduces Amendment 4's to the last digit, and R0 to R4 pass.
- **P13.** `p13_seed_class`, `p13_task_class` and `p13_reading` implement the filed rules, including ties reading
  "not ahead" and the order "unscored, then S too small, then the common class".

One gap between the plan's words and the set: the plan calls D a ranking "without Γ", but D takes its count in each
matrix from S. D is Γ-free in which subcomponents it picks, not in where it looks. Nothing found makes a share, a set or
a verdict wrong.

## 3. Alternatives to "S takes back no more than target-pointing subcomponents"

**Ruled out by the data.**

- *Γ's set beats a direction ranking.* S is never 0.05 ahead of D; the best case is +0.022 on ` which` seed 1.
- *The first-order score is broken.* R6 agrees on 9 or 10 of 10 everywhere. ê(S) and ê(D) land within a factor 1.3
  of ρ.
- *D wins by fitting the training windows.* D is chosen on training windows and scored on held-out ones. Its held-out
  e_c sum matches its restore.
- *S's share comes from interactions among its members rather than from the direction of each one's correction.* The
  sum of S's own first-order effects, ê(S), is within a factor 1.3 of ρ(S) on all four scored states.
- *Amendment 4's win came from matching per layer across matrix types.* Matching per matrix by output change, C_k,
  still takes back less than nothing on emoticon and about 0.10 on ` which`. That closes the Amendment 4 check's
  "mixed matrix types" alternative.

**Weakened but open.**

- *D's emoticon lead measures how much more strongly D's pieces point at the target.* Mostly it does not. On emoticon
  83 and 84 of S's 210 and 213 members have negative training e_c. Their held-out first-order sum is −0.14 and −0.18
  of the gain. S's target-pointing members alone sum to 0.42 and 0.46, against D's 0.45 and 0.51. D excludes
  negative pieces by construction. The emoticon gap is largely S carrying subcomponents that push the other way. That
  is a first-order reading; restoring S's positive part alone was not run.
- *A fully Γ-free ranking does as well.* D borrows S's count per matrix. A global top-|S| by e_c would have at least
  D's first-order sum, but first order fails for the largest pieces, so it was not shown to restore as much.
- *` which` is a real disagreement between seeds.* Both differences are inside the noise floor; a third seed would
  likely read a tie too. Not run.

**Surviving.**

- *S and D are two routes to the same part of the gain, or two separate parts.* D and S are disjoint by
  construction. On ` which`, two disjoint sets of the same per-matrix size each take back about a fifth. S's share sits
  mostly in layer 3, 0.15 and 0.16 of the gain to first order. D spreads more into layers 0 to 2. If restoring S
  together with D took back about 0.4, the switched-on set holds a part of the change that a top-direction ranking
  at this count misses. If it took back about 0.2, they are redundant. This is the strongest surviving alternative to
  "direction suffices". It decides whether ` which`'s tie means "Γ's set is one of several equivalent target-pointing
  sets" or "Γ's set adds something D lacks". Restoring S with D was listed under "Not run".
- *First-order optimal is not best.* D is the best set to first order, not the best set. A ranking that accounts for
  interactions could beat S on ` which` as well, which would turn the tie into "not ahead". This pushes toward the
  run's expected reading, not against it.

**Context for any claim.** S, chosen without the target, is strongly enriched for target-pointing subcomponents. Its
top-direction share is 0.27 to 0.37. If S's members were drawn at random within each matrix, the expected share would
be 0.02 to 0.03, a baseline computed in this check, not filed. S's median member sits at the 79th to 93rd percentile
of its matrix's e_c. Amendment 4 showed a size ranking finds no such set.

## 4. What a claim in the research log may state

The tree reads "outside the tree", so by the plan the log gets a card with the numbers and no further runs. Neither
branch's checked claim applies.

May state: *On the 67M model after a 10-step fine-tune, we ranked subcomponents by their first-order pull on the
target and took the switched-on set's count in each matrix. That set takes back more of the gain than the switched-on
set on emoticon, and about the same on ` which`. So Γ's switched-on set does not take back more than target-pointing
subcomponents do. On emoticon the gap comes mostly from switched-on subcomponents that push against the target. The
switched-on set, found without the target, still holds about ten times the chance share of the most target-pointing
subcomponents.*

May not state: "direction suffices" as the filed reading, since ` which` did not score "not ahead"; that Γ adds
nothing, since on ` which` a target-blind set ties a target-aware first-order ranking; that a ranking with no input from
Γ matches S, since D takes S's count per matrix; that the emoticon gap measures D's pieces pointing more strongly;
that first-order effects add up in general, since they reverse sign for all alive subcomponents on emoticon; whether S
and D are redundant or additive; or anything about ` we` beyond "S too small", where S takes back about a twentieth
and D about a ninth.

## Not run

- Any GPU computation. Restoring S together with D, restoring S's positive-e_c members alone, and restoring a global
  top-|S| set by e_c were not run.
- A third seed on ` which`.
- A second-order or greedy ranking that accounts for interactions.
- A direction ranking matched per layer.
- Positions other than the trigger, other tasks, or the larger model.
