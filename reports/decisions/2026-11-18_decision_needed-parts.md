# Decision: should a small new trigger in front of an old, unchanged circuit count as reuse?

*2026-11-18, evening: the log card was rewritten shorter at the user's request (log v401); this file keeps the fuller text.*

*2026-11-18, definition-lead. A definition question for the user, filed as an ask; you chose to talk it through
(2026-11-18). No recommendation on the main question: it is yours.*

*2026-11-18, later: vocabulary moved to Geiger et al. 2025's (`decisions.md`, "Featurized model replaces
refinement"); content unchanged. A level is now a featurized model, a part a variable, a part's rule its mechanism,
the parts F needs a minimal circuit, the swap a soft intervention from the base model.*

**Measured so far, and whether it was expected.** M is the base model, the original 67M model, and F a fine-tune of
it. The definition says F reuses M, for a behavior under a declared featurizer, to the extent that the variables F
needs still have M's mechanisms on the settings of their parents that occur.
- The score with single neurons as variables (mean mechanism agreement over the circuit F needs) saturated: 0.999 or
  more on every fine-tune in E208, with reuse-minus-twin gaps that follow update size. Not expected; you dropped the
  score from the definition.
- The held-model reading separated only items that are new by construction (a graft 0.13, the rebuilt circuit 0.56)
  from old-task fine-tunes (0.98 or more); matched pairs differed by 0.011 at most. Partly expected.
- What the held cell would add: four hand-built edits whose labels are known (two reuse, two new), read with neurons,
  heads, MLP layers and blocks as variables. Nothing so far has read items with known answers with variables coarser
  than heads.

**The question.** Should a fine-tune that adds a small new trigger in front of an old, unchanged circuit count as
reuse? Example: M copies B after a repeated A through its induction circuit, two attention heads. The "reuse, small"
edit adds eight first-layer neurons that make a new cue A′ look like A; the unchanged heads then copy B. By construction
we call this reuse. As written, the circuit F needs is found by giving every variable M's mechanism and handing F's
mechanism back to as few variables as possible until the behavior returns. A soft intervention on a variable whose
mechanism did not change does nothing, so the induction heads never enter the circuit. The circuit is only the new
front end, whose mechanisms do not agree with M's. So the edit reads "not reused", the same as the edit that writes a
new answer into the same eight neurons. This holds under every featurizer. The definition currently answers "did the
edited variables keep their mechanisms?", not "how much old machinery does the new behavior run through?" (the log box
already says so).

**Options.**
- **(a) Keep the definition as written.** The test runs as planned. Expected: no coarse featurizer separates (the
  three concentrated edits all read 0) and the neuron featurizer reads the thinly spread new edit highest; this mostly
  confirms a derivation. The paper's claim becomes: reuse says whether the variables a fine-tune changed kept M's
  mechanisms, that is, how the change was made. Lost: a re-triggered old circuit reads as new, so "reuse small" cannot
  be told from "new in a few units". The trained pairs are not read under a new featurizer.
- **(b) Count every variable the behavior runs through in F, changed or not.** The unchanged heads then count as
  agreeing with M. The planner's estimate: "reuse, small" reads about 2/3 with blocks as variables, and the new edits
  near 0. The first block, holding the front end, reads 0; the next two, holding the unchanged heads, read 1. So the
  test could pass, and the trained pairs would then be read once under that featurizer.
  This needs a new clause in the definition, in your words, and one more choice that is part of the definition: what
  an unneeded variable is replaced by when testing whether F needs it.
  **Your answer (2026-11-18): if every variable counts, an unneeded variable is replaced by its mean over the
  behavior's inputs.** (The other option was its value on another input, a resample.)

**What the literature does (added 2026-11-18, from the review
`reports/reviews/2026-11-18_review_definition-causal-abstraction.md`, §4 Q1).** Circuit analysis counts every
variable the behavior runs through: sub-circuit analysis (Geiger et al. 2025, §3.5.2) ablates everything outside a
minimal circuit to declared values and asks whether the circuit still produces the behavior. That is option (b). Your
replacement rule, the mean over the behavior's inputs, is standard mean ablation (Wang et al. 2023); resampling (Chan
et al. 2022) is the other standard. This is context, not a recommendation: the main question stays yours.

**A cold read (redteam-lead, 2026-11-18, from `reports/reviews/2026-11-18_review_definition-cold-read.md`).** A reader
who had seen only the definition applied it to reward hacking and emergent misalignment. Under (a), as written, an old
skill aimed at a new cue (reward hacking, emergent misalignment) reads as new: only the variables whose mechanisms
changed are scored. A behavior the base model already has gets no reading, because the set of needed variables is
empty. Under (b), both fine-tuning cases read as mostly reused, as the literature does, but a memorised fact also reads
as mostly reused, since the unchanged recall machinery counts. Neither option names a "created" outcome. The main
question stays yours, with no recommendation.

**A second, smaller question (you said "I don't know"; you may hand it back to the lead).** When we test whether a
variable's mechanism in F agrees with M's, we give the mechanism a setting of its parents and compare its output with
M's mechanism on the same setting. Which settings do we use? Take the example above.
- **(A) What the parents take in F's own run.** F has learned to make the new cue A′ look like A in layer 1, so the
  unchanged induction heads receive "looks like A". M's mechanism for those heads, given the same thing, would also
  copy B. Same output, so the heads read as agreeing. This asks: did the variable's own mechanism change?
- **(B) What the variable took when M read the same text.** M never learned A′, so in M the heads received plain A′
  and did nothing. In F they copy B. The values differ, so the heads read as not agreeing, even though their weights
  never changed. This is activation patching (Prakash et al. 2024). It asks: does the variable take the same value on
  this text as it did in M?

Under B, any reuse that runs an old circuit on a new cue reads as new, because M never ran the circuit on that cue.
Under A, it can read as reuse. **Lead's recommendation: A**, because only A can tell "same machinery, new cue" from
"new machinery". Both are reported either way; the choice sets which one the pass rule uses. It matters only if the
main answer is (b).

**What the review adds to the second question (2026-11-18; review §3 D6 and §4 Q2).** The decided wording already
says "given the same input values, the part's mechanism in F gives the same output as its mechanism in M". That
compares two mechanisms on the same settings of the parents, which is A, the standard reading of "same mechanism" in
causal models. B compares the values a variable takes in two solutions, as activation patching does, so choosing B
would change decided wording. What is really open is whose settings the comparison uses: those that occur in F's
solutions, those in M's, or both pooled. The review suggests pooled, since the standard is the settings that occur on
the data.

**What happens next.** Cell P1.31 (2 runs, due 2026-11-20) is held, and predictions are filed only after your answer.
Under (a), run 1 runs as planned and run 2 most likely does not. Under (b), the plan gains the variable-removal step
with your replacement rule, predictions are filed for it, and the trained pairs are read if a featurizer passes.

**Links.** Plan `reports/plans/2026-11-18_plan_known-answer-levels.md` (points 1 and 2); appendix
`reports/plans/2026-11-18_plan_known-answer-levels_appendix.md` (section 3: the derivation and the 2/3 estimate);
`experiments/known_answer_levels/card.md`; `theory/2026-11-18_level-as-decomposition.md`.

*Names in the reports: cell P1.31 is the known-answer test in the table of work; E208 the neuron reading of 20 saved
fine-tunes; "reading 1/2" are options (a)/(b); "reading A/B" the second question's A/B; "clause 5" the definition's
circuit F needs; older files say "part" for variable, "rule" for mechanism and "level" for featurized model.*
