# Where does the third wording of reuse give no reading?

*2026-11-12, definition-lead, cell P1.18. Reading and writing only: nothing was run for this page. The third wording is for
testing only and is not adopted; `decisions.md` keeps the second wording as the working definition. Every label below
is "in principle": what the wording would say if the measurement came out as the construction or the saved readings
suggest. No verdict is given on any fine-tune. The run that would measure it is cell P1.19, planned in
`reports/plans/2026-11-12_plan_soft-intervention-wording.md`. Revised the same day after an independent critique.*

We read the third wording of reuse, which calls a fine-tune reuse when it is the original model with one of the
original's own variables set by a new rule, against every item the second wording was read on. In principle it labels
the hand-made floor and XOR as new computation, a label the second wording had no way to give.
On the other items the label is set by choices the wording leaves open: the class of allowed rules, how large a
variable may be, and whether a variable named by hand from the task counts as found on the original.
Two hand-made items fail the wording as written, because their variable was taken from the edit itself.

**The wording's own outcomes.** *Reuse*, graded as a pure re-setting or as an old unit with a new trigger; *neither*,
when no proper variable of any causal model found on the original carries the change under the stated class, which the
grade calls a new computation; *no reading*, when the search for that causal model fails.

## The third wording in four lines

These symbols are used throughout.

- $M$ is the original 67M model and $F$ a fine-tune of it, or an edit written into it.
- $H$ is a causal model of $M$ with named variables. $\tau$, the alignment, says which direction of $M$'s activations
  holds each variable's value. The wording requires both to be fixed from $M$ alone, before $F$ is examined.
- $V$ is the variable whose rule, or equation, $F$ changes. $F$ reuses $V$ when $F$ implements $H$ with only $V$'s
  equation replaced, and every other equation of $H$ still holds in $F$ under interchange interventions, not only on
  $F$'s own runs.
- The new equation is graded. A constant, gain or shift of $V$'s own value is a **pure re-setting**. A simple map of
  $H$'s other variables is an **old unit with a new trigger**. Anything more is a **new computation**, not reuse.

The grade depends on a class of allowed maps that the wording leaves open. Three classes are read here, each in words.

- **One-variable class.** The new equation is the old value plus a gain or shift of at most one other named variable of
  $H$. A token enters only as itself, through its own output direction, never through a fitted read of its embedding.
- **Named-variables class.** Any affine map of $H$'s named variables, fitted reads of token embeddings included.
- **Per-token class.** The named-variables class plus any table from an input token to a value, that is, an affine map of
  the token's one-hot code.

## What the items have in common

Every planted edit and every fine-tune on the memorised-fact layout pushes the next-token distribution along one fixed
direction at block 2's input, last position, when a one-bit fact about the last number token is 1. Call that direction's
coordinate $U$. The direction is $Qb$, the constant push trained for the hand-made Always on edit on the plain model's
loss on lowercased text, and loaded unchanged by the later runs. It is trained in `train_always_on` of
`experiments/reuse_variable_test/pieces.py` and loaded in `main` of `experiments/memorised_fact/run.py`. The plain model reads the parity of the last number
exactly along a class-mean direction; call that coordinate $P$. So a natural $H$ has the input tokens, $P$, $U$ and the
rest of the state at that site.
The old matched pairs put a marker after a key or word and train a new answer there. Their natural $V$ is the slice of the
last layer along the answer tokens' unembedding directions, and their trigger is the key, the word, or a property of the
word that the plain model already computes.

## Every item, under the third wording beside the second

"Second wording as read" is the label at the adopted cut-offs of 0.8 and 0.5, with the re-weighting share $G$ and the
rewiring share of the second wording, written $H_2$ here so that it is not confused with the causal model $H$. Seeds are
given seed 0 first.

| Item | $H$ and $\tau$ named on $M$ | $V$ that carries the change | New equation; grade by class | Third wording, in principle | Second wording as read | Gives a reading? |
|---|---|---|---|---|---|---|
| **Planted controls** | | | | | | |
| Planted odd edit | tokens; $P$ along the parity class-mean direction at block 2's input; $U$ along $Qb$; the rest | $U$ | $U \leftarrow U + s\,(P - p_0)$ as intended. New trigger in all three classes. Pure re-setting if $V$ may hold $P$ and $U$ together with a matrix gain | old unit, new trigger, as built | rewired: $G$ 0.63, $H_2$ 0.985 (`experiments/memorised_fact/results/readings.json`) | not exact as built; unchecked. The push direction is not orthogonal to $P$'s, so the edit also moves $P$ and changes $P$'s equation too. The plan rebuilds it with the trigger read one block earlier. $Qb$ raises the hand-naming question below |
| Planted fact edit | as above; the random bit is not a variable of $H$ | $U$ | $U \leftarrow U + s\,L(h)$, $L$ a linear read of the whole site. One-variable: new computation. Named and per-token: new trigger, by the same argument as the memorised fact, since the bit $L$ tracks is a table of the 100 number tokens; the edit's read of the whole site matches that table only as closely as a fit shows | new computation under one-variable; new trigger under the other two, to the fit | rewired: $G$ 0.48, $H_2$ 0.98 (same file) | yes in principle, with the fit deciding how near under the last two classes; the hand-naming question as above |
| Planted floor | as above | $U$ | $U \leftarrow U + s\,r(a,b)$, a random bit per ordered pair of the last two numbers. No class here holds it: it is neither a table of one token nor linear in the site's 768 directions over the pairs | new computation in all three classes, for this $H$ | no map found: $G$ $-0.15$, $H_2$ 0.00 (same file) | yes in principle, for this $H$; the hand-naming question as above |
| Always on, hand-made | $U$ along $Qb$, the four-direction push this edit trained | $U$ | $U \leftarrow U + b$, a shift. Pure re-setting in all classes | pure re-setting, by construction | re-weighted: $G$ 0.92, $H_2$ 0.98 (`experiments/reuse_variable_test/results/rules.json`) | no reading as written: $U$ is the edit's own trained push, not fixed from $M$ alone. A reading only if that clause is relaxed |
| Cue, hand-made | $C$, "the cue is in the window", along the ridge read at block 1's input; $U$ along $Qb$ | $U$ | $U \leftarrow U + w\cdot h_1 + a$: a gain of $C$. New trigger in all classes, if $H$ names $C$ | old unit, new trigger, by construction | rewired: $G$ 0.67, $H_2$ 0.96 (same file) | yes, if $C$ is named. $Qb$ was trained for Always on, not read from this edit: the hand-naming question |
| XOR, hand-made | the tokens; $U$ along $Qb$ | $U$ | $U \leftarrow U + \mathrm{xor}$ of four per-token bits. A four-way interaction: no class holds it, and the plain model reads it at chance | new computation in all classes, for this $H$ | no map found: $G$ $-0.03$, $H_2$ $-0.01$ (same file) | yes, as for Cue |
| Table, hand-made | the last token's embedding; the span the table writes at block 2's input, up to 33 directions | that span | $V \leftarrow V + C\,e(t) + c$, a full matrix on the embedding of 32 keys. One-variable: new computation. Named and per-token: new trigger, since 32 keys fit in 768 directions | depends on the class | rewired: $G$ $-0.03$, $H_2$ 0.99 (same file) | no reading as written: the span is the edit's own trained map. If that clause is relaxed, only if the alignment may hold 33 directions |
| **Memorised fact and twin** | | | | | | |
| Memorised fact, 2 seeds | as for the planted edits, at each of four sites | $U$ if the fine-tune routes its push through $Qb$; not known | if so, $U \leftarrow U + \alpha\,\mathrm{fact}(t)$. One-variable: new computation. Named and per-token: new trigger, since 100 number tokens fit in 768 directions | set by the class, not the fine-tune | no map found: $G$ 0.10, 0.12; $H_2$ 0.36, 0.37 (same file as the planted edits) | not measured. $Qb$ is fixed by the task's target before training, not read from $F$: the hand-naming question |
| Twin, last number odd, 2 seeds | as above | $U$, as above | if so, $U \leftarrow U + \alpha P$: a gain of $P$. New trigger in all classes, if the fine-tune reads $M$'s own $P$; an interchange on $P$ tests that | old unit, new trigger, if the interchange on $P$ moves the push | no map found: $G$ 0.39, 0.40; $H_2$ 0.47, 0.46 (same file) | not measured, as for the fact |
| **Old matched pairs**, 2 seeds a side | | | | | | |
| In-context copy against lookup | the earlier token $B$, which the plain model's copying heads carry; the key $A$; the answer slice of the last layer | the answer slice | copy: $V \leftarrow V + \gamma\,e(B)$, a gain of a named variable, so new trigger in all classes. Lookup: a table of 32 keys, so new computation under one-variable and new trigger under the other two | copy: new trigger. Lookup: set by the class | rewired and rewired: $G$ 0.70, 0.69 against 0.33, 0.29 (`experiments/old_task_labels/results/labels_cut_0p8_0p5.json`) | only if the answer slice, spanned by a pool of answer tokens, fits the alignment class, and only if a slice of the last layer counts as a variable |
| Copy against cipher, 24 keys | the key's embedding; the answer slice | the answer slice | copy: a gain of the key's own embedding, new trigger everywhere. Cipher: a table of 24 keys, new computation under one-variable only | copy: new trigger. Cipher: set by the class | rewired and rewired: $G$ 0.68, 0.68 against 0.39, 0.33 (same file) | as above; the slice has 24 directions |
| Identity against random map, 32 keys | as above | as above | as above, with 32 keys | as above | rewired and rewired: $G$ 0.69, 0.71 against 0.28, 0.26 (same file) | as above; 32 directions |
| Real case label against its twin | capitalisation of the word before the marker, a class-mean direction; the label slice, one direction from label A to label B | the label slice | real: a gain of capitalisation, new trigger everywhere, unless $M$'s own label slice already varies with capitalisation. Twin, a label per word: new computation under one-variable; under named, its 800 words are just over the 768 directions, so an affine read is close but not exact and the fit decides; per-token: new trigger | real: new trigger, or pure re-setting if $M$'s slice already tracks capitalisation. Twin: set by the class and the pool size | re-weighted and rewired: $G$ 0.83, 0.83 against 0.52, 0.55 (same file) | yes on one direction, if a slice of the last layer counts as a variable |
| Real repeat label against its twin | "this word occurred earlier", a class-mean direction; the label slice | the label slice | real: a gain of that variable, new trigger everywhere, with the same proviso as the case label. Twin, a label per word over 400 words: new computation under one-variable, new trigger under the other two | real: new trigger. Twin: set by the class | rewired and rewired: $G$ 0.76, 0.74 against 0.54, 0.57 (same file) | as for the case label |

**Reading it.** Under the one-variable class every control and every pair comes out as constructed, with each reuse
side an old unit with a new trigger and each new side a new computation. The one exception is the hand-made Table,
built as a new connection from old parts, which this class calls a new computation. That agreement comes from how $H$ was
written: we named the reuse side's trigger as a variable because we knew the task. Under the other two classes both sides
of every per-token pair read new trigger, the planted fact edit reads new trigger to within a fit, and the floor and the
XOR stay new computation. Each new-computation label holds for the one $H$ named here. The wording's *neither* also
needs that no other $H$ found on $M$ carries the change.

A conditional behaviour, one shown on some inputs and not others, can still be a pure re-setting. A gain on $V$'s own old
value turns on where $M$'s $V$ was already large. So "a conditional behaviour reads a new trigger" holds only where
$M$'s $V$ does not vary with the condition. This page assumes that for $U$ along $Qb$ and for the answer and label slices,
on the reading that $M$ never acts along them, and never measured it.

## Where it gives no reading

Each cause below names the items it hits.

**The one $H$ named here may not carry the change.** The third wording's test is that $F$ implements $H$ with only $V$'s
equation replaced: every other equation of $H$ still holds in $F$ under interchange interventions, not only on $F$'s own
runs. A fine-tune changes every weight and may write its push at several sites, so the $H$ named here, with $U$ along one
direction at one site, may fail that test. On the second wording's two swaps, a different test, the best one-site
variable closed at most 0.37 of the fact's gap and 0.47 of the twin's. That hints one direction will not be enough.
If the test fails for this $H$, the wording does not say *neither*, which needs that no proper variable of any $H$ found
on $M$ carries the change. It says *no reading*, and that comes from naming one $H$ by hand rather than searching, not
from the wording. Hits the memorised fact and twin, likely the old pairs.

**"Every other equation still holds" needs a bar, and holds only on the inputs tried.** A fine-tune never leaves the rest
of the network exactly unchanged, so "holds" means "holds to an interchange bar", which the wording leaves unset. On a
closed input set the check covers swaps between inputs of that set and nothing else. A value no input produces is never
tried. Hits every fine-tune. The planted edits were built to change only $U$'s equation, but the planted odd edit's push
also moves $P$, and whether the others hold was never checked.

**With full next-token outputs, "M implements H" is nearly empty.** The wording follows Geiger et al., whose high-level
model has a symbolic output. Reading $M$'s whole next-token distribution as $H$'s output is this project's measurement
choice, not the wording's. Under that choice the output equation of $H$ has to be $M$'s own remaining layers applied to
$H$'s variables and to the rest of the state. Then any split of a site into directions is an exact abstraction of $M$, and
the interchange test on $M$ constrains only how each named variable is computed from the tokens. Hits every item, as read
here.

**The rest of the state is a variable too.** If the rest of the state at a site may be $V$, every edit that adds a
constant at that site, like Always on, is a shift of that rest and so a pure re-setting. An edit that adds a read of $P$,
like the planted odd edit, is not a re-setting of the rest, since $P$ lies outside it. If the rest may appear in the map,
any linear read of the site is a simple map, and the planted fact edit reads new trigger exactly. The wording does not
say whether the rest counts. Hits the planted fact edit and Always on.

**"Fixed from M alone" is written in the wording.** $H$ and $\tau$ are fixed from $M$ alone, before $F$ is examined. Always
on's $U$ is the push that edit itself trained, and the Table's span is that edit's own map. As written, the wording gives
these two no reading; a reading needs the clause relaxed. For every other item on the memorised-fact layout, $U$ along
$Qb$ was fixed before the item's edit or fine-tune existed and is not read from it, so nothing written fails. But $Qb$
was trained on the plain model's loss for one task, not found by a search on $M$, and the old pairs' answer slices are
named from the task's answer tokens. Whether a variable named by hand from the task counts as fixed from $M$ alone is a
separate question the wording does not settle. Hits Always on and Table as written; raises the question for every
other item.

**A slice of the output is a proper part.** "Proper part" excludes the whole output and a whole layer. A single direction
of the last layer, from label A to label B, is a proper part, since the other directions still move the output. So a
push on the output can always be written as an intervention on that slice, and the wording cannot tell it from a new
edge into the output's equation. On 8-token inputs this is where every old pair's change lands. Hits the old pairs.

**Token identity is a variable, so per-token tables are affine.** Inputs are variables of every $H$. A table from one
token to a bit is affine in the token's one-hot code, and also linear in its embedding whenever the table covers at most
768 tokens. That holds for the 100 numbers, the 24 and 32 keys and the 400 repeat words. So the memorised fact, the
planted fact edit to within a fit, the cipher, the random map, the lookup and the repeat twin read new trigger unless the
class excludes tables. The case twin has 800 words, just over the width, so a fit bar decides it. The map class decides
the label, not the model. Hits the fact, the planted fact edit and every new side.

**A variable may bundle the trigger.** If $V$ may hold several directions with a matrix gain, a trigger placed inside
$V$ turns a new trigger into a pure re-setting. The planted odd edit becomes a pure re-setting of the pair of $P$ and
$U$. This is the second wording's four-direction limit in a new place. Hits the planted odd edit, the twin, Cue.

**The alignment is too small.** The Table writes up to 33 directions and the key tables' answers span 24 or 32. At four
directions or fewer there is no reading for them. Hits Table and the three key-table pairs.

**The twin and the fact take identical values on the inputs.** On these inputs "the last number is odd" is itself a table
of the last token. Values alone cannot say whether the twin reads $M$'s parity variable or memorised the table. Only an
interchange on $P$ inside $F$ can. For the fact there is no named variable to intervene on, so "which variable does it
read" has no reading there. Hits the fact and twin.

## What changes from the second wording

Each line says what the third wording does on these items; none is a recommendation.

- A new computation gets a label of its own. Given a fixed variable, the best map in a class is a least-squares fit with an
  exact optimum, so "no map in the class fits" is a measurement, not a search limit. The floor and the XOR would read new
  computation for the $H$ named here, where the second wording read "no map found".
- The planted odd edit is read through an $H$ fixed on $M$, not through a search; the second wording's search reached 0.63
  of it. As built, the edit also moves $P$, so its third-wording reading is not exact either, and that is unchecked.
- The planted pair separates under the one-variable class only: the odd edit reads new trigger, the fact edit new
  computation. Under the other two classes both read new trigger. The second wording's re-weighting share put them 0.14
  apart.
- It can ask whether the twin uses $M$'s own parity variable, by an interchange on $P$ inside $F$. The second wording's
  reversion does not name the variable a fine-tune reads.
- The real case label, the second wording's only re-weighted old task, reads new trigger here unless $M$'s own label
  slice already varies with capitalisation, which was not measured.
- The grade has three steps and no share. The reuse side's higher re-weighting share, seen in all five old pairs and the
  fact, has no counterpart unless a fit share is reported beside the grade.
- On per-token pairs the label is set by which variables $H$ names, by the map class, and by the pool size against the
  model's width, all fixed before the fine-tune is read.
- It needs these stated in advance: the variables of $H$, the alignment, the map class, the interchange bar, the input
  domain.
- Where the test fails for the one $H$ named, a fine-tune gets no reading, where the second wording gave a share on every
  item it read.

## Open for the user

Each of these is the user's to decide; no recommendation is made.

- Which class grades the new equation: a gain of one other variable, any affine map of named variables, or per-token
  tables too?
- Are input tokens variables of $H$, and as embeddings or as one-hot codes?
- May the rest of the state at a site be $V$, or appear in the map?
- What interchange bar counts as "M implements H", and as "every other equation still holds in F"? Read on what output:
  the full next-token distribution, or a symbolic answer as in Geiger et al.?
- Which alignments are allowed: one direction per variable, up to four, or the span of an answer set? One site or several?
- Does a slice of the last layer along output tokens count as a variable, or as the output?
- Does a variable named by hand from the task, such as the push $Qb$ or an answer slice, not found by a search on $M$,
  count as fixed from $M$ alone?
- Should Always on and the Table, whose variable is the edit's own, stay without a reading, or is the clause relaxed for
  hand-made controls?
- May $V$ hold several directions with a matrix gain, so that a trigger inside $V$ counts as re-setting?
- Should a fit share be reported beside the three-step grade?

## Departures from the literature

- Geiger et al. 2025 give the high-level model a symbolic output and score interchange accuracy on output labels. Here the
  output is a next-token distribution, so $H$'s output equation is $M$'s own remaining layers; that is why "M implements H"
  constrains so little. This is our measurement choice, not the third wording's.
- Distributed alignment search learns each variable's direction by interchange training on the low-level model's outputs.
  Here a variable $M$ barely acts on, such as parity, is aligned by class means instead, a linear read, and the push $Qb$ is
  taken from a trained edit.
- Treating a fine-tune as a soft intervention on its base is the wording's own step, not from a paper.

## Not run

- Nothing. Every label above is in principle, from construction or from saved second-wording readings.
- No interchange check, carry, reversion or map fit on any item. That is cell P1.19.
- No search for $H$ on $M$ by interchange training; every $H$ here is named by hand.
- No check that the planted odd edit's trigger and push directions are orthogonal, or of how far the edit moves $P$.
- No measurement of whether $M$'s own value along $Qb$, or its answer and label slices, varies with each item's trigger.
- No change to the third wording, the second wording or `decisions.md`.

*Sources: third wording `~/commute-data/scratch/definition-session/2026-11-12_reuse-as-soft-intervention_v3.md`; second
wording `.../2026-11-09_reuse-at-a-variable_v2.md`; gate page `reports/analyses/2026-11-10_analysis_gate-page-working-definition.md`;
memorised fact `reports/analyses/2026-11-12_analysis_memorised-fact.md`, `experiments/memorised_fact/results/readings.json`,
`experiments/memorised_fact/run.py`; hand-made pairs `reports/analyses/2026-11-10_analysis_reuse-variable-first-test.md`,
`experiments/reuse_variable_test/results/rules.json`, `experiments/reuse_variable_test/pieces.py`; old pairs
`experiments/old_task_labels/results/labels_cut_0p8_0p5.json`; constructions `src/commute/planted.py`,
`src/commute/variable_read.py` (`piece_edit`), `src/commute/old_tasks.py`; repeat pool
`reports/plans/2026-11-06_plan_e125-fresh-reuse-pair_appendix.md`. Numbers ledger beside this file.*
