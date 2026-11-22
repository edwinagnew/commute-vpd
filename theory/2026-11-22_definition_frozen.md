# The definition of reuse: sentence and exact form

*FROZEN by the user, 2026-11-22: "freeze the sentence and exact form". Sections 1 and 2 are the definition. Only the
user changes them. Red-teamed the same day by a GPT-6.2 Sol reviewer against the text of Geiger et al.; its verdict
was "freeze after wording fixes", and the fixes that keep the content are applied below (section 7 lists them and the
one that was declined). Definition and section numbers are from Geiger et al. 2025, arXiv 2301.04709 version 4; the
JMLR numbering differs and the paper must use one of the two throughout.*

## 1. The sentence

A fine-tune F reuses the base model M, for a behavior, to the extent that the variables F needs for that behavior
still have M's mechanisms, on the settings of their parents that occur.

## 2. The exact form

**Setting.** M and F are neural networks with one architecture. A causal featurization is declared, fixed from M
alone, and applied unchanged to both. It specifies one shared signature (the input variables, a set U of intermediate
variables, the output variables, and their value sets) and determines the complete mechanisms of each featurized
network, including on settings that ordinary inputs do not reach. Intermediate variables take values in real vector
spaces. In the sense of Geiger et al., Def. 5, each mechanism is a total function of the signature's settings. For
each intermediate variable u we write both mechanisms, f_u^M and f_u^F, as functions of the union of u's parent sets
in M and in F, each ignoring the arguments it does not depend on.

- *Coverage* (user, 2026-11-18). Every mechanism the fine-tune changed belongs to some intermediate variable; a
  remainder variable covers what the features do not. The output variables have the same mechanisms in both models.
- *A coordinate M lacks.* If the signature has a coordinate with no counterpart in the base model, its zero padding is
  part of the declared featurization: its mechanism in M is identically zero.
- *Determinate runs.* Every ordinary run on D, and every single-variable interchange tested below, has a unique
  solution. Acyclic featurized models, which feed-forward networks give, meet this.

A behavior B is a partial function from inputs to designated outputs (Section 3.5.2) that F computes on its domain D.
Which outputs are designated (for example logits or decoded tokens) is part of declaring B.

**Settings that occur.** For an input x in D, let p_u(x) be the joint setting of u's parents in the unmodified run of
the featurized F on x. P_u = {p_u(x) : x in D}.

**Kept.** A variable u is kept when F's mechanism is M's mechanism up to one positive output gain on every setting
that occurs:

    u is kept  iff  there is c > 0 such that f_u^F(p) = c * f_u^M(p) for every p in P_u.

For the same c, this equality holds exactly when, for every x in D, the original full setting of F on x remains a
solution after the soft intervention (Def. 10) that replaces only f_u^F by c * f_u^M. If that intervened model has a
unique solution for each x, the intervention leaves the full setting unchanged.

**Needed.** A variable u is needed for B when transplanting its value alone between two runs can change the
designated output: there are x and x' in D such that running the featurized F on x, with u set to the value it takes
on x' (a single-variable interchange intervention, Def. 49), gives a designated output other than B(x). K is the set
of needed variables.

**Reuse.** The reused variables are R = {u in K : u is kept}, and the new variables are K \ R. Reuse is full among
the needed intermediate variables when R = K, and absent among them when R is empty and K is not. When K is empty,
full reuse holds vacuously and does not assert that F has an unchanged circuit for B. "New" means needed and not
kept; it does not assert that the variable was absent in M.

**What is declared.** Besides M and F, the inputs to the exact form are the declared causal featurization and the
behavior B, including its domain and designated outputs. No distribution, distance, fitting procedure or ablation
baseline enters it.

## 3. What is not frozen: the graded form

"To the extent" is made a number by a reading. A reading declares, before any result, a similarity between F's and
M's mechanisms at a variable, a distribution over the settings compared, and a statistic that aggregates over settings
and variables. This flexibility is analogous to Geiger et al. (Def. 45, Section 3.2), where the similarity between
resulting settings, the distribution over interventionals and the aggregation statistic are "points of variation" to
"be filled in based on the use case". It is an analogy: a reading compares local mechanisms and is not automatically
an instance of Def. 45.

An admissible reading is defined on a declared class of cases and has a specified upper endpoint; on every case in
that class it attains that endpoint if and only if R = K, including when K is empty. Readings belong to the methods
and results, not to the definition. Two are on file: the share of the circuit by squared cosine (compares whole
outputs) and the write share (compares the change with M's mechanism). Whether either meets the admissibility
condition has NOT been checked (see section 6).

## 4. Properties of the exact form

1. **Do-nothing.** If F = M then every variable is kept with c = 1, so reuse is full (possibly vacuously).
2. **Gain.** If each mechanism of F is M's mechanism multiplied by a fixed positive constant (1 where untouched),
   every variable is kept. Downstream variables see different parent values, but their mechanisms, as functions, are
   M's. This is about the declared mechanisms, not about arbitrary rescalings of weights.
3. **A coordinate M lacks.** If F gives a zero-padded coordinate a nonzero value on some setting that occurs, it is not
   kept. If it is also needed, reuse is not full.
4. **New is possible.** There exist admitted cases with R empty and K not empty (u_M = x, u_F = x + 1, Y = u,
   D = {0, 1}). This need not hold at every featurization.
5. **Total.** For every admitted pair of featurized models and behavior, K and R are uniquely determined, including
   when K is empty. The gain c need not be unique; the sets are. This does not guarantee that a graded reading is
   defined or computable.
6. **Looks inside.** Two fine-tunes with the same input-output behavior can have different R (M: u = x, Y = u;
   F1: u = 2x, Y = u/2 keeps u; F2: u = 2x + 1, Y = (u - 1)/2 does not). The same weight update applied to two bases
   can also give different verdicts. So the definition is a function neither of behavior alone nor of the update alone.
7. **Relative to the featurization.** If the whole real-vector output is one intermediate variable with an identity
   readout, and F's output is not constant on D, then K is that variable, and it is kept exactly when F's output is one
   positive multiple of M's throughout D. Finer descriptions can change the result in either direction. A claim of
   reuse is a claim at a declared description.

## 5. Departures from the literature

- Geiger et al. develop causal transformations between causal models, including same-granularity bijective
  translations (Def. 30). Ours is a reuse predicate comparing two trained networks at one declared signature; it is
  not a claim of exact transformation in their sense.
- We fix the causal featurization from M and apply it unchanged to F. Geiger et al. describe modular features as
  intervention algebras accessed by bijective translations (Sections 3.1, 3.6; Def. 30); a featurization is such a
  translation only when it meets those requirements. The paper does not use the noun "featurizer".
- Sameness up to one positive output gain is ours (user, 2026-11-18).
- Comparing on the parent settings that occur in F's own runs on D is ours.
- Our "needed" uses a single-variable interchange intervention (Def. 49) with both inputs in D; the existential
  definition of need is ours. Geiger's sub-circuit analysis (Section 3.5.2) instead asks whether a circuit's edges
  suffice to preserve the behavior when the other connections are ablated to specified values.

## 6. Known limits (to state in the paper)

- **Exact proportionality is a stringent condition.** How often trained fine-tunes meet it is an empirical question.
  Hand-built edits anchor the exact endpoints; a declared graded reading reports a separate quantity.
- **"Needed" is single-site sensitivity on D, not necessity.** It misses a variable that is constant on D (g = 1,
  Y = g*x), and variables needed only jointly (three copies of x with Y their median: K is empty). It can include a
  dispensable copy (u = v = x, Y = max(u, v)). Sub-circuit analysis has the complementary failures: ties between
  minimal circuits, and dependence on the ablation value.
- **Not invariant to a lossless change of coordinates.** With h_M = x, h_F = 2x, Y = h, the variable is kept; in the
  coordinate h' = h + 1 it is not. A gain is relative to the origin of the declared coordinates. Fixing the
  featurization from M guards against choosing an alignment that favours F, not against this dependence.
- **Scope of the verdict.** "Kept" can hold on parent settings M itself never reached, and ignores differences outside
  F's runs on D. So the exact form alone does not show that M already performed the behavior, nor how training
  arrived at F. On a one-input D, K is empty for every F.
- **Readings.** Admissible readings can disagree below the top: on one hand-built new edit the two on file give 0.98
  and about 0.015. A pointwise squared cosine cannot enforce one gain across settings or exclude a negative gain, so
  the saved readings are not yet shown to be admissible. This needs a $0 check of what they compute.
- **Memorization is not a separate criterion.** Its exact classification is relative to the declared behavior and
  featurization; any graded score is reported separately (user, 2026-11-22: not a design criterion).

## 7. What the red team changed, and what was declined

Applied (wording, content unchanged): mechanisms typed on the union of parent sets; the featurization must determine
full mechanisms and unique runs; the soft-intervention paraphrase; the endpoint wording for K empty; narrower
attributions to Geiger; the admissibility sentence; properties 4, 5 and 7 restated; the claim "gradient training
moves every weight" removed as not generally true.

Declined: the reviewer proposed replacing the sentence in section 1 with a longer one, because "fully reuses M" can
overstate. The sentence is the user's and stays; the qualification is carried by the endpoint wording in section 2.

Review text: session record of 2026-11-22 (top-level session). Working page with attacks, defences and the history:
https://claude.ai/artifact/J9RSzb7m9horKiTNTqFnvo
