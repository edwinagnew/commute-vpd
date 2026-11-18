# Decision: should a small new trigger in front of an old, unchanged circuit count as reuse?

*2026-11-18, definition-lead. A definition question for the user, filed as an ask; you chose to talk it through
(2026-11-18). No recommendation on the main question: it is yours.*

**Measured so far, and whether it was expected.** M is the original 67M model, F a fine-tune of it. The definition
says F reuses M, for a behaviour at a declared level, to the extent that the parts F needs still compute what M's parts
compute on the same inputs.
- The neuron-level score (mean rule retention of the parts needed) saturated: 0.999 or more on every fine-tune in E208,
  with reuse-minus-twin gaps that follow update size. Not expected; you dropped the score from the definition.
- The held-model reading separated only items that are new by construction (a graft 0.13, the rebuilt circuit 0.56)
  from old-task fine-tunes (0.98 or more); matched pairs differed by 0.011 at most. Partly expected.
- What the held cell would add: four hand-built edits whose labels are known (two reuse, two new), read at the neuron, head,
  MLP-layer and block levels. Nothing so far has read items with known answers at levels coarser than heads.

**The question.** Should a fine-tune that adds a small new trigger in front of an old, unchanged circuit count as
reuse? Example: M copies B after a repeated A through its induction circuit, two attention heads. The "reuse, small"
edit adds eight first-layer neurons that make a new cue A′ look like A; the unchanged heads then copy B. By construction
we call this reuse. As written, the parts F needs are found by giving every part M's rule and handing F's rule back to
as few parts as possible until the behaviour returns. An unchanged part is not moved by that swap, so the induction
heads never enter the set. The set is only the new front end, which does not keep M's rule. So the edit reads "not
reused", the same as the edit that writes a new answer into the same eight neurons. This holds at every level and under
every decomposition. The definition currently answers "did the edited parts keep their rules?", not "how much old
machinery does the new behaviour run through?" (the log box already says so).

**Options.**
- **(a) Keep the definition as written.** The test runs as planned. Expected: no coarse level separates (the three
  concentrated edits all read 0) and the neuron level reads the thinly spread new edit highest; this mostly confirms a
  derivation. The paper's claim becomes: reuse says whether the parts a fine-tune changed kept M's rules, that is, how
  the change was made. Lost: a re-triggered old circuit reads as new, so "reuse small" cannot be told from "new in a few
  units". The trained pairs are not read at a new level.
- **(b) Count every part the behaviour runs through in F, changed or not.** The unchanged heads then count as kept.
  The planner's estimate: "reuse, small" reads about 2/3 at the block level, and the new edits near 0. The first
  block, holding the front end, reads 0; the next two, holding the unchanged heads, read 1. So the test could pass, and
  the trained pairs would then be read once at that level.
  This needs a new clause in the definition, in your words, and one more choice that is part of the definition: what an
  unneeded part is replaced by when testing whether F needs it.
  - *Its mean over the behaviour's inputs:* removes what the part carries about this input, keeps its average; simple,
    but the mean can be a value the part never takes.
  - *Its value on another input (resample):* every value is one the network really produces; noisier, and it needs a
    rule for which other input.

**A second, smaller question (lead's recommendation).** When a part is given M's rule, which input does it see?
(A) Its own input in F's run, so only the rule is replaced. Recommended: only under A can an untouched re-used
circuit read as kept. (B) The value it had when M ran the same text, as in activation patching (Prakash et al. 2024).
Under B an unchanged circuit after a re-trigger reads "not kept", because it never fired in M. Both are reported either
way; this picks the one the pass rule uses. It matters only once the main question is answered.

**What happens next.** Cell P1.31 (2 runs, due 2026-11-20) is held, and predictions are filed only after your answer.
Under (a), run 1 runs as planned and run 2 most likely does not. Under (b), the plan gains the part-removal step with
your replacement rule, predictions are filed for it, and the trained pairs are read if a level passes.

**Links.** Plan `reports/plans/2026-11-18_plan_known-answer-levels.md` (points 1 and 2); appendix
`reports/plans/2026-11-18_plan_known-answer-levels_appendix.md` (section 3: the derivation and the 2/3 estimate);
`experiments/known_answer_levels/card.md`; `theory/2026-11-18_level-as-decomposition.md`.

*Names in the reports: cell P1.31 is the known-answer test in the table of work; E208 the neuron-level reading of 20
saved fine-tunes; "reading 1/2" are options (a)/(b); "reading A/B" the second question's A/B; "clause 5" the
definition's "parts F needs".*
