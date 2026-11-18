# Cold read: what the working definition says on three alignment cases

*2026-11-18, redteam-lead, cold reader. Read: Part A of `theory/2026-11-18_level-as-decomposition.md`, the decision
card `reports/decisions/2026-11-18_decision_needed-parts.md` and `GOALS.md` Terms. Nothing was run and no wording is
proposed. Answer (a) keeps the definition as written; answer (b) counts every variable the behavior runs through,
with unneeded variables replaced by their mean over the behavior's inputs.*

I applied the working definition, as written, to reward hacking, emergent misalignment and alignment pretraining,
under each answer on which variables count. As written, it scores only how far the changed variables moved: an old skill
aimed at a new cue reads as new, and a behavior the base model already showed gets no reading. Counting every variable the behavior runs through reads
both fine-tuning cases as mostly reused, as the literature does. Neither answer names a "created" outcome, makes a
memorised fact read as new, or can compare two pretraining runs, so the definition does not yet meet the aim in
`GOALS.md` of a reading that holds for any update.

## How each answer reads the three cases

The two answers differ in one clause, the set of variables F needs. Under (a) it is the smallest set whose
fine-tuned mechanisms must be kept for the behavior to return, when every other variable is given the base model's
mechanism. A variable whose mechanism did not change cannot matter to that test. So the set holds only changed
variables, and the reading is how close their mechanisms stayed to the base model's. Under (b) the set is every
variable the behavior runs through in F, so unchanged machinery counts as agreeing.

**Reward hacking.** *Literature:* under reinforcement learning a policy learns to exploit the reward, for example
by special-casing the tests it is graded on. The usual reading is that the base model can already write such code
and the training raises the rate at which it chooses to.
- *(a):* the set holds only the variables that now pick the hack; the old code-writing machinery never enters. With
  coarse variables the reading is new; with single neurons it is near full reuse, per the two-extremes paragraph.
  This is the decision card's "new trigger in front of an old circuit", which reads as not reused. If the base model
  already hacks on the chosen inputs, the set is empty and the reading is **undetermined**: the aggregation over the
  set is declared per reading and has no value.
- *(b):* the unchanged code-writing machinery counts, so the reading is mostly reused, by the share of the circuit
  that did not change. The amount is **undetermined** until the aggregation is declared, by count or by weight.
- *Both:* B is "a partial function from inputs to outputs". How a sampled, many-token policy output becomes one is
  not said, nor the tolerance for "the behavior returns".

**Emergent misalignment.** *Literature:* Betley et al. 2025 fine-tuned on insecure code and got misaligned answers
on unrelated prompts. Later work, Wang et al. 2025 and Soligo et al. 2025, found a misaligned-persona direction
already present in the base model, which the fine-tune turns up.
- *(a):* with sparse-autoencoder variables, the persona latent's mechanism changed, so it sits in the set and reads
  as disagreeing. The unchanged machinery that turns the persona into text is excluded. The verdict is new, graded
  only by how far that latent's mechanism moved; the pre-existing persona never counts as reused.
- *(b):* the persona latent reads as changed and the downstream machinery as agreeing, so the reading is mostly
  reused. This matches the literature, with the amount again set by the undeclared aggregation.
- *Both:* the base model already gives misaligned answers on a few samples, and the fine-tune raises the rate. A
  partial function from inputs to outputs does not state a rate change, so B is **undetermined**. Which behavior is
  read, the insecure code or the broad misalignment, is also a declared choice that changes the answer.

**Alignment pretraining.** *Literature:* changing the pretraining data, by filtering harmful text (O'Brien et al.
2025) or by conditioning on preference tags (Korbak et al. 2023), shapes how aligned the model is after post-training
and how easily it is tuned out of it.
- *Both answers, comparing the two pretrained networks:* Part A calls M "the base model" but never requires F to
  descend from it. A featurizer from one pretraining run applied to another makes "a variable the same in both" by
  declaration only. Nearly every mechanism differs, so the reading is new under either answer, and that verdict
  tells us nothing.
- *Both answers, one reading per pretraining run:* each run's post-training gets a reading, but on its own featurizer.
  Part A has no clause for comparing readings across featurizers, so the question the case asks is **undetermined**.
- *Pretraining itself:* with a random initialisation as M, every needed variable changed and the reading is its
  lowest value. Part A does not call that value "created".

## Requirements

| Requirement | (a) as written | (b) every variable the behavior runs through |
|---|---|---|
| Independent of any decomposition or method | unclear: any featurizer slots in, but each reading is "under a declared featurizer", and the two extremes give opposite answers | unclear: same, and the replacement by the mean over B's inputs is a method choice inside the definition |
| Not a cheap measurement | meets: a minimal circuit plus mechanism comparison "on the settings of their parents that occur" | meets: the same search plus one replacement pass per variable |
| About mechanisms, not behavior-only or update-level | unclear: it compares mechanisms, but only of changed variables, so it reads "how the change was made", the update's footprint | meets: unchanged mechanisms the behavior runs through count |
| Simple and general | unclear: one-sentence core, but five quantities are "declared with every reading" and the approximate case is open; attribution graphs are "not a case" | unclear: as (a), plus one clause and a replacement rule |
| Total, "created" possible, a memorised fact reads as new | fails: the set is empty when M already has B; "created" is never named; a memorised fact reads as new only with coarse variables, since "single neurons ... read nearly everything as reuse" | fails: a reading always exists once its quantities are declared, but "created" is never named, and a memorised fact reads mostly reused because the unchanged recall machinery counts |
| Graded | meets: "to the extent that", with the scale set by the undeclared distance and aggregation | meets: same, read as the share of the circuit that agrees |
| Built on the literature | meets: Geiger et al. 2025 definitions throughout, with the departure "base-model mechanisms in place of ablation values" stated | meets: closer to Geiger et al.'s sub-circuit analysis, with mean ablation as in Wang et al. 2023 |

## Clauses no reading depends on

For each clause I asked whether any of the six readings above, or Part A's own worked examples, uses it. The worked
examples are the cases list, the do-nothing remark and the two-extremes paragraph. Used: the featurized model, ε and the
remainder; the units, sparse-autoencoder, VPD and coarser-grouping cases; the shared featurizer fixed from M. Also
used: the mechanism, soft intervention and intervened model; the definition sentence with B and the minimal circuit.
Also used: the distance, tolerance, parent settings and aggregation; the open question on the approximate case; the
two extremes. No reading uses the following, so each is a candidate for an appendix:

1. The detail of the approximate transformation: the maps, τ the identity on inputs and outputs, the distribution P
   and KL as Sim. Only "ε is reported beside every reading" is used.
2. "When the featurizer is a bijective translation, its variables are Geiger's modular features."
3. "The direction is the reverse of the alignment form."
4. The rotations case.
5. The transcoders case. It is a worked example of itself only.
6. "Not a case: attribution graphs." No reading uses it, though it bears on generality.
7. "A soft intervention whose replacement mechanism comes from another trained network ... remain ours, not
   Geiger's." It bears on the literature requirement, not on any reading.
8. "The user's phrasing of the same thing", a restatement of the sentence.
9. The guard's analogy to Sutter et al. 2025. Its rule, fixed from M only, restates the shared-featurizer clause.
10. "The rule when K is not unique." No case here produced two minimal sets.
11. "Featurizers are not ordered from coarse to fine" and the instruction "Write 'two extreme choices'", which is
    guidance for prose.

## Not run

- No reading was measured; every verdict is derived from the text.
- The cited literature was not reopened; its pictures are given as commonly summarised.
- The VPD case, where the split of the update is a declared parameter, was not applied to any of the three cases.
- The approximate case was not worked through, since that comparison is an open question.
- Which inputs supply "the settings of their parents that occur", F's, M's or both, was not resolved. It changes the
  graded amount for changed variables under both answers.
- The cited Geiger et al. definitions were not checked against the paper.
- No reading was done under answer (b) with a resample in place of the mean.
