# Analysis: which subcomponent numbers tell a new mechanism from an existing one? (round 7)

*2026-10-20, coding agent. Plan: `reports/plans/2026-10-20_plan_new-vs-existing-signatures.md`. Predictions:
`experiments/new_vs_existing/results/predictions.json`, filed before the run and scored as filed. Results:
`experiments/new_vs_existing/results/{tasks,new_vs_existing}.json`; figures in `experiments/new_vs_existing/figures/`;
the explanation is `notebooks/E12_new_vs_existing.ipynb`. Provenance at the end.*

We taught the model two things it can already do, repeating a word, and two things it cannot, an arbitrary word-to-word
mapping, and looked for any per-subcomponent number that tells the two apart. Nothing about the weight update does. It
is equally expressible by the subcomponents, it lands on the same kind of subcomponents, and two training seeds agree
equally. The difference sits entirely in how the frozen importance function reads the trained model. For repeating a
word it switches on about 120 existing subcomponents, which then carry the behaviour. For the mapping it switches on
about 35, and the behaviour runs through subcomponents it keeps off.

## What we were trying to test

Every round so far asked whether a map from the fine-tune's update to the decomposition passes VPD's checks. This round
steps back and asks what the decomposition can see about the update at all. We taught the model two kinds of thing in
one format and looked for any per-subcomponent number that tells them apart.

The format is the key → value task of round 6. A Pile window of 41 tokens has one token replaced by a rare marker and
the next by a common word, the key. The model is fine-tuned to predict a particular word after that pair. In the two
*copy* tasks the word is the key itself: "marker the" is followed by "the". The model has copying machinery, so this
strengthens something it can do. In the two *mapping* tasks the word is a fixed random other word, the same 32 pairs
every time: "marker the" is followed by "copy", say. Nothing in the model does this. Each task teaches 32 pairs in one
fine-tune. The second task of each class uses a different set of 32 keys, so within each class the two tasks differ in
words, not in rule, and the only designed difference between the classes is the rule.

Each task gets a full-weight fine-tune of the 24 decomposed matrices, run at two seeds: eight fine-tuned states. A state
is scored at its first checkpoint that has learned the task, meaning held-out probability of the target at least one
half. The step and the size of the update are recorded beside every number. Nothing else is trained: Γ, Δ and the
released subcomponents are as published.

The numbers, per state. T is the hundred alive subcomponents the lift moves most, where the lift writes the update as
the least-norm change $(dU, dV)$ of the alive subcomponents with $dU V^\top + U dV^\top \approx \delta W$.

1. **How much of the update the subcomponents cannot express.** The share of the update's input-weighted energy that
   lies outside every alive subcomponent's read and write directions, divided by what a random update of the same
   per-matrix size reads. A value of 1 means the update is no more expressible than noise. A random update is run
   through the same number as the check that the null is 1.
2. **How much of the learned behaviour is left in Δ after the lift.** Of the fine-tune's gain in the target's log-odds
   at the trigger, the share the lifted state carries only through Δ. Leaving the update in Δ reads 1 by construction.
3. **Were the moved subcomponents doing the job already?** In the original model, remove T's rank-1 matrices from the
   weights and read how much the target's log-odds at the trigger fall. The reference is five random sets of the same
   size per matrix, drawn from the same decile of causal importance at the trigger. The ratio of T's drop to the random
   sets' mean is 1 when T is no more relevant than chance. The same ablations in the fine-tuned model say whether T carries
   the trained behaviour.
4. **Which route the behaviour takes.** The route split of the lifted state at the trigger. The re-switching share r is
   the part carried because the frozen Γ switches existing subcomponents on for the new activations. The switched-off
   share is the part carried by subcomponents Γ keeps switched off. Round 6 read 0.80 and 0.52 for these on the identity
   and the random map, the one separation already seen.
5. **Do two seeds move the same subcomponents?** The Jaccard overlap of the two seeds' T sets, against the overlap of
   two random sets of the same size.

Reported without a filed rule: T's mean causal importance at the trigger before and after the update, the lift energy
sitting on subcomponents switched off at the trigger, and how often T is on over Pile. Also the overlap of T and of
the switched-on set with the subcomponents most active where the model naturally predicts the current token again, Γ
coverage of the task gradient, and the sizes of the switched-on and switched-off sets.

## What we expected, and why

Suppose the decomposition were a complete story of the model. Then an existing mechanism would show as a set of
subcomponents that are already on and relevant at W₀, that the gradient lies in, that the update lands on, and that
two seeds agree on. A new mechanism would show as none of those, with the behaviour having nowhere to go but the switched-off
subcomponents or Δ. So:

- Number 1 separates, copy tasks lower: expected at most 0.7 against at least 0.85.
- Number 2 does **not** separate. Round 6 read 0.13 for the identity and 0.18 for the random map.
- Number 3 separates, copy tasks higher: expected at least 3 against between 0.5 and 2. For the random mappings the
  target's probability at W₀ is about one in a million, so a ratio near 1 is also what no signal reads.
- Number 4 separates on both shares, as in round 6. It is the round's positive control.
- Number 5 separates, copy tasks higher, at least 0.5.

A number separates when the two classes' ranges do not overlap and the gap between them exceeds the larger
within-class range.

## What we observed

Held fixed throughout: Γ, Δ, the released subcomponents, the window format, the fine-tune recipe. Varied: the rule, copy or mapping; the key set; the seed.

### 1. Which of the five numbers separates the classes? Only the route

Figure: `figures/separation.png`. One marker per fine-tuned state, both classes on each panel, references dashed.

| task | class | seed | step | held-out P | ‖δW‖/‖W₀‖ | off-target KL | inexpressible / null | random δW | left in Δ | relevance ratio | r | e_gate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ident32 | copy | 0 | 10 | 0.957 | 0.0136 | 0.188 | 0.621 | 0.996 | 0.128 | 41.2 | 0.802 | 0.058 |
| ident32 | copy | 1 | 10 | 0.962 | 0.0139 | 0.180 | 0.632 | 0.995 | 0.116 | 35.5 | 0.761 | 0.113 |
| ident32b | copy | 0 | 10 | 0.977 | 0.0139 | 0.205 | 0.600 | 0.996 | 0.137 | 23.3 | 0.791 | 0.061 |
| ident32b | copy | 1 | 10 | 0.945 | 0.0141 | 0.195 | 0.619 | 0.995 | 0.160 | 53.5 | 0.804 | 0.026 |
| rand32 | mapping | 0 | 10 | 0.586 | 0.0151 | 0.693 | 0.606 | 0.996 | 0.184 | 19.7 | 0.273 | 0.519 |
| rand32 | mapping | 1 | 10 | 0.546 | 0.0153 | 0.655 | 0.627 | 0.995 | 0.182 | 5.2 | 0.266 | 0.528 |
| rand32b | mapping | 0 | 30 | 0.996 | 0.0254 | 0.097 | 0.588 | 0.996 | 0.241 | 4.2 | 0.289 | 0.445 |
| rand32b | mapping | 1 | 30 | 0.988 | 0.0260 | 0.102 | 0.595 | 0.995 | 0.229 | 10.7 | 0.279 | 0.469 |

Seed overlap of T: ident32 0.786, ident32b 0.724, rand32 0.681, rand32b 0.724; two random sets 0.015–0.026. The
direct share d is 0.012–0.020 and the interaction share within 0.003 of zero on every state.

**Reading it.** Only the route separates. The re-switching share is 0.76 to 0.80 for the copy tasks and 0.27 to 0.29
for the mappings. The switched-off share is 0.03 to 0.11 against 0.45 to 0.53. Round 6's values for the identity and the 32-pair random map sit inside each cluster. The first copy state is round
6's identity run reproduced to three digits. The update is equally expressible in both classes, about 0.6 of the random
null for every state, and a random update of the same size reads 1.0. The share left in Δ is a little higher for the
mappings, but the gap of 0.02 is below the within-class spread of 0.06, as filed. The moved subcomponents are far
more relevant than chance in both classes; the copy tasks' ratios spread from 23 to 54 and the mappings' from 4 to 20,
and the gap is a tenth of the spread. Two seeds of a task move the same subcomponents about equally often in both
classes.

### 2. Where does the update land, and were those subcomponents doing the job already? The same place in both classes

Figures: `figures/landing.png`, `figures/relevance.png`.

| task | seed | T mean CI at trigger, W₀ | T mean CI, trained | lift energy on switched-off | T on-rate over Pile | T ∩ switched-on | T energy share | drop of target at W₀, T | drop, random sets | KL at trigger, T | KL, random | drop in fine-tuned model, T |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ident32 | 0 | 0.178 | 0.184 | 0.957 | 0.138 | 0.023 | 0.214 | 6.27 | −0.14 … 0.19 | 10.8 | 0.43–0.75 | 18.7 |
| ident32 | 1 | 0.175 | 0.184 | 0.957 | 0.138 | 0.025 | 0.219 | 5.88 | −0.38 … −0.02 | 10.5 | 0.34–0.70 | 18.7 |
| ident32b | 0 | 0.170 | 0.174 | 0.958 | 0.134 | 0.021 | 0.202 | 7.86 | −0.82 … −0.06 | 12.5 | 0.39–0.58 | 21.1 |
| ident32b | 1 | 0.183 | 0.201 | 0.957 | 0.144 | 0.030 | 0.202 | 5.89 | −0.22 … 0.05 | 10.9 | 0.41–0.53 | 19.0 |
| rand32 | 0 | 0.179 | 0.186 | 0.949 | 0.133 | 0.014 | 0.241 | 1.60 | −0.19 … 0.07 | 11.2 | 0.34–0.62 | 14.0 |
| rand32 | 1 | 0.173 | 0.175 | 0.953 | 0.130 | 0.007 | 0.260 | 1.64 | −0.49 … 0.57 | 11.0 | 0.43–0.55 | 13.5 |
| rand32b | 0 | 0.168 | 0.187 | 0.967 | 0.129 | 0.033 | 0.216 | 2.39 | −0.83 … −0.47 | 12.4 | 0.42–0.65 | 20.6 |
| rand32b | 1 | 0.172 | 0.193 | 0.968 | 0.130 | 0.034 | 0.214 | 1.53 | −0.37 … 0.03 | 11.9 | 0.36–0.50 | 19.6 |

Mean on-rate over all alive subcomponents on 64 Pile rows: 0.015. T by matrix, every state: the layer-1 and layer-3
MLP input matrices hold 55 to 65 of the hundred. Γ coverage of the task gradient at W₀ is 0.24 for both copy tasks
and 0.22 for both mappings.

**Reading it.** The update lands in the same place for both classes, and that place is busy rather than specific. The
hundred subcomponents the lift moves most have a mean causal importance of about 0.18 at the trigger, before and after
training, in all eight states. More than 95% of the lift's energy sits on subcomponents whose importance at the trigger
is below one half. These subcomponents are on at about 13% of Pile positions, nine times the alive average, so the
update goes to subcomponents the model uses everywhere. Removing them from the original model changes the next-token
distribution at the trigger by 10 to 12 nats, against half a nat for a random set of matched importance. The copy
target falls by 6 to 8 nats. The mapping target falls by 1.5 to 2.4 nats, so even the arbitrary target was supported
by these subcomponents before training, which no mechanism story predicts. In the fine-tuned model, removing them takes
away the whole gain in both classes, and also wrecks the distribution. The relevance number as designed cannot tell
"the mechanism for this behaviour" from "important at this position in general".

### 3. How does the frozen Γ respond to the trained model? Differently

Figure: `figures/gamma.png`.

| task | seed | switched on | switched off | mean abs. CI change over alive | seed overlap of switched-on | switched-on on-rate over Pile | layer 0 / 1 / 2 / 3 |
|---|---|---|---|---|---|---|---|
| ident32 | 0 | 123 | 5 | 0.026 | 0.797 | 0.061 | 7 / 23 / 60 / 33 |
| ident32 | 1 | 107 | 5 | 0.024 | | 0.062 | 5 / 20 / 54 / 28 |
| ident32b | 0 | 139 | 9 | 0.028 | 0.847 | 0.060 | 7 / 20 / 60 / 52 |
| ident32b | 1 | 138 | 9 | 0.028 | | 0.061 | 8 / 19 / 59 / 52 |
| rand32 | 0 | 41 | 0 | 0.015 | 0.533 | 0.088 | 1 / 3 / 6 / 31 |
| rand32 | 1 | 51 | 0 | 0.016 | | 0.081 | 3 / 5 / 7 / 36 |
| rand32b | 0 | 27 | 1 | 0.019 | 0.655 | 0.093 | 3 / 3 / 5 / 16 |
| rand32b | 1 | 21 | 1 | 0.018 | | 0.105 | 3 / 2 / 5 / 11 |

For the copy tasks the layer-2 members are mostly the attention key, query and output matrices, 17 to 20 each; the
layer-3 members are the MLP input and output matrices. For the mappings the layer-3 members are the MLP input matrix,
8 to 15, and the attention output matrix. The overlap of every set with the top hundred "repeat the current token"
subcomponents on Pile is 0.00 to 0.01, chance; that control had 289 such positions in 32,768 and has no power.

**Reading it.** Here the classes part. For the copy tasks Γ switches on 107 to 139 subcomponents at the trigger, for
the mappings 21 to 51, and almost nothing is switched off in either class. The two seeds of a copy task switch on sets
that overlap by about 0.8, the two seeds of a mapping by about 0.6. The switched-on subcomponents of the copy tasks sit
mainly in the layer-2 attention matrices and the layer-3 MLP, where a copying mechanism would plausibly live; those of
the mappings sit almost only in layer 3. These switched-on subcomponents are not the ones the update moved: their
overlap with T is 0.03 or less in every state. So for strengthening, a diffuse weight change shifts the activations
enough for Γ to switch on a coherent, pre-existing set, and that set carries the behaviour. For creating, Γ finds much
less to switch on, and the behaviour is carried by subcomponents it leaves off.

## What we learned

For the goal's second branch, "new capability" turns out to be a statement about how Γ reads the trained model, not
about the weight update written in the subcomponent basis. On every number that reads the update itself the two
classes are indistinguishable, and the update lands on relevant, generically busy subcomponents either way. What
differs is whether the frozen Γ, shown the trained activations, finds a large existing set to switch on. That is the
mechanism round 6 called reveal, now with a within-class floor behind it. It also says why every frozen-Γ map has
struggled: the information that separates the two cases is not in δW projected onto the subcomponents, it is in how
Γ responds, and a map that freezes Γ cannot read it. Updating Γ, which the backlog has carried since the start, is the
next thing a map needs.

## How compelling is the evidence

Two tasks and two seeds per class, all in one format and at one update size. The route separation is as wide as in
round 6 and its within-class spread is 0.04, so it is robust here. The three nulls are nulls, not reversals: the
predicted direction held for the relevance ratio and nearly for the residual share, but the within-class spread
swallowed the gap. One confound: the second mapping task learned at step 30 with an update twice the size of the
others. Its numbers match the first mapping task at step 10, so the class signature does not depend on step or size
in this range. The first mapping task only just learned at step 10, with held-out probability 0.55 to 0.59. The
relevance check is blind to the difference between "the mechanism for this" and "important at this position". The
"repeat the current token" control over Pile read chance for every set and had 289 positions, so it says nothing. The
expressibility number is the input-weighted share pooled over the matrices whose subcomponents do not already span
every direction, as in rounds 4 and 5.

## Surprises

Three of the five filed separations did not happen. We predicted that the copy tasks' updates would be more
expressible, that the subcomponents they move would be more relevant at the original model, and that their seeds would
agree more. None of those held: the update looks the same on every update-side number. The larger surprise is that
the subcomponents the update moves most are relevant at the trigger in both classes, even for a target the model
gives one in a million. They are also almost disjoint from the subcomponents Γ switches on. The mechanism story
we wrote down, "the update lands on the mechanism", is wrong for both classes: the update lands on the busy
subcomponents, and Γ does the rest.

## What next

- Read T by relative change instead of absolute energy, and at a smaller size such as 20. If a relative T overlaps the
  switched-on set for the copy tasks, the "busy subcomponents" reading was an artefact of the energy ranking.
- Ask what the switched-on subcomponents of the copy tasks do in the original model, on Pile positions where the model
  copies. An attention-based copying mechanism would predict high importance there, and a positive answer would make
  the layer-2 attention set the first named mechanism in this project.
- A map that lets Γ move, scored by the weak version on these four tasks. If the mapping tasks then pass where the
  frozen-Γ maps fail, the "brittle" branch is answered for this format.

## Scorecard

| id | claim | rule | read | verdict |
|---|---|---|---|---|
| G1 | every fine-tune learns | held-out P ≥ 0.5 within 300 steps, all 8 | 6 at step 10, rand32b at step 30 | pass |
| G2 | the lift is exact | lifted state matches the fine-tune's held-out P to 1e-3, shares sum to 1 | worst difference 1e-5 | pass |
| G3 | the null reads 1 | random δW's inexpressible ratio in [0.9, 1.1], all 8 | 0.995–0.996 | pass |
| P1 | update more expressible for copy tasks | separates, existing lower | 0.60–0.63 vs 0.59–0.63 | FAIL (null) |
| P2 | residual share does not separate | no separation | 0.12–0.16 vs 0.18–0.24; gap 0.02 < within 0.06 | pass |
| P3 | moved subcomponents relevant only for copy tasks | separates, existing higher | 23–54 vs 4–20; gap 3.6 < within 30 | FAIL (direction held) |
| P4 | route separates | r and e_gate both separate | r 0.76–0.80 vs 0.27–0.29; e_gate 0.03–0.11 vs 0.45–0.53 | pass |
| P5 | seeds agree more for copy tasks | separates, existing higher | 0.72–0.79 vs 0.68–0.72 | FAIL (null) |

## Not run

- A third class the model half has, such as a plural rule, to see whether the switched-on count grades with how much of
  the mechanism exists.
- T by relative change, and sizes other than 100; CI-matched sets at other deciles.
- Pile faithfulness of these states; not the question this round.
- Seed-to-seed overlap at matched held-out probability for the first mapping task.
- A ladder task as a third "existing" format.
- The fine-tunes' δW files were written to the pod's container disk and not downloaded before it was terminated; the
  plan had assumed cluster storage. Re-creating them is a nine-minute run.

**Provenance.** `experiments/new_vs_existing/results/new_vs_existing.json` sha256 41c5adcbbe2204267595af406c1ea227279658cc0c50f7197878f9eb89924d25;
`tasks.json` 4e1635224663bec3c266183398bf26d30f41b25bb74e620eb58ad6ea2a5050e0. RunPod secure pod `commute-new_vs_existing`,
NVIDIA A40, smoke 5 min 52 s, scored run 8 min 49 s, total $0.24. Cluster job 26045 was submitted and cancelled by the user
before it started. Logs `cluster/logs/commute-nve-pod-{smoke,scored}.out`. Code: `experiments/new_vs_existing/run.py`,
`src/commute/signatures.py`; Codex adversarial review of the diff approved with no findings.
