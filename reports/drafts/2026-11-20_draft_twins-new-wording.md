# Draft: a fact memorised through small changes reads as new

*definition-lead, 2026-11-20, for the user's approval. Nothing here is adopted; no live file is edited. You decided
(2026-11-20 ~19:15, answer "2b") that the memorising twins stay a case the reuse score R must read low on, so the
definition must say such a fact is new.*

## 1. What says the opposite today

- Paper, `writeup-lead/paper/paper/main.tex` lines 337-338 (Properties): "Cost: a memorised fact also reads as mostly
  reused when it is stored by small changes in an otherwise old circuit."
- The same sentence: `reports/drafts/2026-11-18_draft_sec4-bullets-ask48b.md` line 89, `GOALS.md` line 75,
  `theory/2026-11-18_definition_consolidated.md` line 115, and the log's definition box
  (`docs/research_log/parts/sections/overview.html` line 130).
- Its cause is the grade, not the Properties line: draft line 68 ("let $s_u$ grade how far
  $\tilde F^{(1)}_u = \tilde F^{(0)}_u$ on $\mathcal{V}_u$") with line 75 (direction only, by squared cosine). A small
  weight change leaves every mechanism's output nearly the same, so every $s_u$ is near 1, however much the behavior
  depends on the change. Deleting line 89 alone would make the Properties list wrong, not the definition right.

## 2. Candidate rewordings

**A. Kept means "M's mechanism can stand in for it" (changes the sentence).** A set of variables of the circuit
keeps M's mechanisms when the soft intervention (Geiger Def. 10) that gives all of them M's mechanisms, up to an
output gain each, still preserves the behavior within the same α as the circuit. R is the largest such share of the
circuit.
- *Changes:* the meaning of "still have M's mechanisms" in the definition's sentence; the grade by squared cosine
  becomes one reading, not the definition. The Properties line flips: the variables a memorised fact is stored in
  cannot take M's mechanisms without losing it, so they read as new however small the change; untouched variables
  in the circuit still read as kept. A new cue on an old circuit, a behavior M already has and the gain rule read as
  today.
- *Readings:* all change and must be re-read: R, the six known-answer edits (the thinly spread new edits should now
  read new, the ten-fold gain edit kept) and the nine pairs. Old R numbers stand as readings of the squared-cosine
  grade. Close to the held model's survival, but per variable inside the circuit rather than the whole network.
- *Total:* yes. Giving no variable M's mechanism always preserves the behavior, so a share always exists.

**B. The same, one variable at a time (changes the sentence).** $s_u$ is the share of the behavior kept when $u$
alone is given M's mechanism, up to an output gain.
- *Changes and readings:* as A, and cheaper to read, but a fact spread thinly over many variables, each change
  removable on its own, reads as reused again. That is the case the twins test, so B fails it by construction.
- *Total:* yes.

**C. Keep the sentence, add a clause.** "A behavior whose input-output map M computes nowhere, on any setting of its
variables that occurs, is new, however small the weight change."
- *Changes:* only the Properties line and one new clause; R as read stands wherever the clause does not fire.
- *Readings:* the twins' memorised sides and the two built new edits read 0 if the clause fires; the reuse items,
  and most of the nine pairs, are unchanged. But the clause needs its own test of "computes nowhere", and a mostly
  old behavior with one memorised exception jumps from near 1 to 0.
- *Total:* only once "computes nowhere" has a test that gives yes or no on every fine-tune; none exists today.

## 3. Recommendation: A

A is the simplest that is defensible: it changes one word's meaning instead of adding a rule, uses only Geiger's
soft intervention and the α the circuit already declares, and stays total. It makes the memorised parts new without
a test for memorisation. Proposed wording (sentence, then the new Properties line):

> A fine-tune F reuses the base model M, for a behavior, to the extent that M's mechanisms, each up to an output
> gain, can stand in for those of the variables the behavior runs through and still preserve the behavior.

> A memorised fact reads as new where it is stored, however small the changes that store it: M's mechanisms cannot
> stand in for those variables without losing it.

Not run: no reading under A exists yet; reading it needs a new cell from you.
