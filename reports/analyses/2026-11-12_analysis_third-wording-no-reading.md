# Where does the third wording of reuse give no reading?

*2026-11-12, definition-lead, cell P1.18. Reading and writing only: nothing was run for this page. The third wording is for
testing only and is not adopted; `decisions.md` keeps the second wording as the working definition. Every label below
is "in principle": what the wording would say if the measurement came out as the construction or the saved readings
suggest. No verdict is given on any fine-tune. The run that would measure it is cell P1.19, planned in
`reports/plans/2026-11-12_plan_soft-intervention-wording.md`.*

We read the third wording of reuse, which calls a fine-tune reuse when it is the original model with one of the
original's own variables set by a new rule, against every item the second wording was read on. In principle it reads
the planted and hand-made controls as they were constructed, and it can positively read a new computation, which the
second wording never could.
But the memorised fact and every new side of an old pair get the label the user's choice of allowed rules sets, not one
the fine-tune shows, and fine-tunes get no reading when no single fixed variable carries their change.

## The third wording in four lines

These symbols are used throughout.

- $M$ is the original 67M model and $F$ a fine-tune of it, or an edit written into it.
- $H$ is a causal model of $M$ with named variables. $\tau$, the alignment, says which direction of $M$'s activations
  holds each variable's value. Both are fixed from $M$ alone, before $F$ is looked at.
- $V$ is the variable whose rule, or equation, $F$ changes. $F$ reuses $V$ when $F$ is $M$ with only $V$'s equation
  replaced, and every other equation of $H$ still holds in $F$ under interchange interventions, not only on $F$'s own runs.
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
coordinate $U$. The plain model reads the parity of the last number exactly along a class-mean direction; call that
coordinate $P$. So a natural $H$ found on $M$ has the input tokens, $P$, $U$ and the rest of the state at that site.
The old matched pairs put a marker after a key or word and train a new answer there. Their natural $V$ is the slice of the
last layer along the answer tokens' unembedding directions, and their trigger is the key, the word, or a property of the
word that the plain model already computes.

## Every item, under the third wording beside the second

"Second wording as read" is the label at the adopted cut-offs of 0.8 and 0.5, with the re-weighting share $G$ and the
rewiring share of the second wording, written $H_2$ here so that it is not confused with the causal model $H$. Seeds are
given seed 0 first.

| Item | $H$ and $\tau$ found on $M$ alone | $V$ that carries the change | New equation; grade by class | Third wording, in principle | Second wording as read | Gives a reading? |
|---|---|---|---|---|---|---|
| **Planted controls** | | | | | | |
| Planted odd edit | tokens; $P$ along the parity class-mean direction at block 2's input; $U$ along the push; the rest | $U$ | $U \leftarrow U + s\,(P - p_0)$, exact. New trigger in all three classes. Pure re-setting if $V$ may hold $P$ and $U$ together with a matrix gain | old unit, new trigger, by construction | rewired: $G$ 0.63, $H_2$ 0.985 (`experiments/memorised_fact/results/readings.json`) | yes, exactly. Caveat: the push direction is not orthogonal to $P$'s, so the edit also moves $P$; the plan rebuilds it with the trigger read one block earlier |
| Planted fact edit | as above; the random bit is not a variable of $H$ | $U$ | $U \leftarrow U + s\,L(h)$, $L$ a linear read of the same site. One-variable and named: new computation. Per-token: about a table of the last number, so the fit decides | new computation under the first two classes; open under per-token | rewired: $G$ 0.48, $H_2$ 0.98 (same file) | yes under the first two classes. Under per-token, a fit bar the user sets decides |
| Planted floor | as above | $U$ | $U \leftarrow U + s\,r(a,b)$, a random bit per ordered pair of the last two numbers. No class here holds it: it is neither a table of one token nor linear in the site's 768 directions over the pairs | new computation in all three classes, by construction | no map found: $G$ $-0.15$, $H_2$ 0.00 (same file) | yes: the first built item that any wording here reads positively |
| Always on, hand-made | $U$ along the trained push, a four-direction subspace at block 2's input | $U$ | $U \leftarrow U + b$, a shift. Pure re-setting in all classes | pure re-setting, by construction | re-weighted: $G$ 0.92, $H_2$ 0.98 (`experiments/reuse_variable_test/results/rules.json`) | yes, if $H$ may name a direction $M$ never uses for anything; that direction was trained, not found on $M$ |
| Cue, hand-made | $C$, "the cue is in the window", along the ridge read at block 1's input; $U$ | $U$ | $U \leftarrow U + w\cdot h_1 + a$: a gain of $C$. New trigger in all classes, if $H$ names $C$ | old unit, new trigger, by construction | rewired: $G$ 0.67, $H_2$ 0.96 (same file) | yes, if $C$ is named |
| XOR, hand-made | the tokens; $U$ | $U$ | $U \leftarrow U + \mathrm{xor}$ of four per-token bits. A four-way interaction: no class holds it, and the plain model reads it at chance | new computation in all classes, by construction | no map found: $G$ $-0.03$, $H_2$ $-0.01$ (same file) | yes |
| Table, hand-made | the last token's embedding; the span the table writes at block 2's input, up to 33 directions | that span | $V \leftarrow V + C\,e(t) + c$, a full matrix on the embedding of 32 keys. One-variable: new computation. Named and per-token: new trigger, since 32 keys fit in 768 directions | rewired by construction under the second wording; under the third it depends on the class | rewired: $G$ $-0.03$, $H_2$ 0.99 (same file) | only if the alignment may hold 33 directions. At four or fewer: no reading |
| **Memorised fact and twin** | | | | | | |
| Memorised fact, 2 seeds | as for the planted edits, at each of four sites | $U$ if the fine-tune routes its push through the fixed direction; not known | if so, $U \leftarrow U + \alpha\,\mathrm{fact}(t)$. One-variable: new computation. Named and per-token: new trigger, since 100 number tokens fit in 768 directions | set by the class, not the fine-tune | no map found: $G$ 0.10, 0.12; $H_2$ 0.36, 0.37 (same file as the planted edits) | not measured. Likely none: the second wording's best one-site variable closed at most 0.37 of the gap |
| Twin, last number odd, 2 seeds | as above | $U$, as above | if so, $U \leftarrow U + \alpha P$: a gain of $P$. New trigger in all classes, if the fine-tune reads $M$'s own $P$; an interchange on $P$ tests that | old unit, new trigger, if the interchange on $P$ moves the push | no map found: $G$ 0.39, 0.40; $H_2$ 0.47, 0.46 (same file) | not measured. Likely none on one direction at one site, as for the fact |
| **Old matched pairs**, 2 seeds a side | | | | | | |
| In-context copy against lookup | the earlier token $B$, which the plain model's copying heads carry; the key $A$; the answer slice of the last layer | the answer slice | copy: $V \leftarrow V + \gamma\,e(B)$, a gain of a named variable, so new trigger in all classes. Lookup: a table of 32 keys, so new computation under one-variable and new trigger under the other two | copy: new trigger. Lookup: set by the class | rewired and rewired: $G$ 0.70, 0.69 against 0.33, 0.29 (`experiments/old_task_labels/results/labels_cut_0p8_0p5.json`) | only if the answer slice, spanned by a pool of answer tokens, fits the alignment class, and only if a slice of the last layer counts as a variable |
| Copy against cipher, 24 keys | the key's embedding; the answer slice | the answer slice | copy: a gain of the key's own embedding, new trigger everywhere. Cipher: a table of 24 keys, new computation under one-variable only | copy: new trigger. Cipher: set by the class | rewired and rewired: $G$ 0.68, 0.68 against 0.39, 0.33 (same file) | as above; the slice has 24 directions |
| Identity against random map, 32 keys | as above | as above | as above, with 32 keys | as above | rewired and rewired: $G$ 0.69, 0.71 against 0.28, 0.26 (same file) | as above; 32 directions |
| Real case label against its twin | capitalisation of the word before the marker, a class-mean direction; the label slice, one direction from label A to label B | the label slice | real: a gain of capitalisation, new trigger everywhere. Twin, a label per word: new computation under one-variable; under named, its 800 words are just over the 768 directions, so an affine read is close but not exact and the fit decides; per-token: new trigger | real: new trigger, never pure re-setting. Twin: set by the class and the pool size | re-weighted and rewired: $G$ 0.83, 0.83 against 0.52, 0.55 (same file) | yes on one direction, if a slice of the last layer counts as a variable |
| Real repeat label against its twin | "this word occurred earlier", a class-mean direction; the label slice | the label slice | real: a gain of that variable, new trigger everywhere. Twin, a label per word over 400 words: new computation under one-variable, new trigger under the other two | real: new trigger. Twin: set by the class | rewired and rewired: $G$ 0.76, 0.74 against 0.54, 0.57 (same file) | as for the case label |

**Reading it.** Under the one-variable class every control and every pair comes out as constructed, with each reuse
side an old unit with a new trigger and each new side a new computation. The one exception is the hand-made Table,
built as a new connection from old parts, which this class calls a new computation. That agreement comes from how $H$ was written:
we named the reuse side's trigger as a variable because we knew the task. Under the other two classes both sides of every
per-token pair read new trigger, and the floor and the XOR stay new computation. No conditional behaviour, the real case
label included, can read pure re-setting, because a behaviour shown on some inputs and not others needs a trigger.

## Where it gives no reading

Each cause below names the items it hits.

**No fixed variable carries the change.** A fine-tune changes every weight and may write its push at several sites.
The third wording then needs one variable whose swap from $F$ into $M$ moves $M$ to $F$, and whose swap back reverts $F$.
On the second wording's searches, the best variable at one site closed at most 0.37 of the fact's gap and 0.47 of the
twin's. If a fixed direction carries no more, both fine-tunes read no reading. This is not evidence of new computation:
the wording says a failed search for $H$ is no reading. Hits the memorised fact and twin, likely the old pairs.

**"Every other equation still holds" needs a bar, and holds only on the inputs tried.** A fine-tune never leaves the rest
of the network exactly unchanged, so "holds" means "holds to an interchange bar", which the wording leaves unset. On a
closed input set the check covers swaps between inputs of that set and nothing else. A value no input produces is never
tried. Hits every fine-tune; the planted edits hold exactly by construction.

**With full next-token outputs, "M implements H" is nearly empty.** The output equation of $H$ must reproduce a whole
distribution, so it has to be $M$'s own remaining layers applied to $H$'s variables and to the rest of the state. Then
any split of a site into directions is an exact abstraction of $M$. The interchange test on $M$ constrains only how
each named variable is computed from the tokens. Hits every item.

**The rest of the state is a variable too.** If the rest of the state at a site may be $V$, every edit at that site is a
re-setting of that rest. If it may appear in the map, any linear read of the site is a simple map, and the planted fact
edit reads new trigger. The wording does not say whether the rest counts. Hits the planted fact edit and Always on.

**A behaviour the original never shows has no $H$ to find.** The plain model never pushes along the planted direction
and never outputs the cipher's answers. An interchange search on $M$'s outputs cannot find a variable $M$ does not use, so
$U$, the answer slice and the push are named by hand from the task. The parity variable is read by class means, because
$M$ reads parity exactly but barely acts on it. Whether naming by hand meets "fixed from M alone" is the user's call.
Hits every item.

**A slice of the output is a proper part.** "Proper part" excludes the whole output and a whole layer. A single direction
of the last layer, from label A to label B, is a proper part, since the other directions still move the output. So a
push on the output can always be written as an intervention on that slice, and the wording cannot tell it from a new
edge into the output's equation. On 8-token inputs this is where every old pair's change lands. Hits the old pairs.

**Token identity is a variable, so per-token tables are affine.** Inputs are variables of every $H$. A table from one
token to a bit is affine in the token's one-hot code, and also linear in its embedding whenever the table covers at most
768 tokens. That holds for the 100 numbers, the 24 and 32 keys and the 400 repeat words. So the memorised fact, the
cipher, the random map, the lookup and the repeat twin read new trigger unless the class excludes tables; the case twin
has 800 words, just over the width, so a fit bar decides it. The map class decides the label, not the model. Hits the fact and every new side.

**A variable may bundle the trigger.** If $V$ may hold several directions with a matrix gain, a trigger placed inside
$V$ turns a new trigger into a pure re-setting. The planted odd edit becomes a pure re-setting of the pair of $P$ and
$U$. This is the second wording's four-direction limit in a new place. Hits the planted odd edit, the twin, Cue.

**The alignment is too small.** The Table writes up to 33 directions and the key tables' answers span 24 or 32. At four
directions or fewer there is no reading for them. Hits Table and the three key-table pairs.

**The twin and the fact take identical values on the inputs.** On these inputs "the last number is odd" is itself a table
of the last token. Values alone cannot say whether the twin reads $M$'s parity variable or memorised the table. Only an
interchange on $P$ inside $F$ can. For the fact there is no named variable to intervene on, so "which variable does it
read" has no reading there. Hits the fact and twin.

## Where it does better than the second wording, and where worse

**Better.**
- A new computation can be read positively. Given a fixed variable, the best map in a class is a least-squares fit with an
  exact optimum. "No map in the class fits" is then a measurement, not a search limit. The floor and the XOR read built
  by construction.
- The exact planted edit's problem goes away. With $H$ and $\tau$ fixed on $M$, the planted odd edit's readings are exact
  by construction, where the second wording's search reached only 0.63 of it.
- The planted pair separates under the one-variable and named classes: the odd edit is a new trigger, the fact edit a new
  computation. The second wording's share put them only 0.14 apart.
- It can ask whether the twin uses $M$'s own parity variable, by an interchange on it. The second wording's reversion could
  not name which variable a fine-tune reads.

**Worse.**
- No conditional behaviour can be a pure re-setting. The real case label, the second wording's only re-weighted old task,
  becomes a new trigger like its twin's reuse siblings.
- The grades are three steps, with no share. The one result that held in all five old pairs and the fact, the reuse side's
  higher re-weighting share, has no counterpart unless a fit share is reported beside the grade.
- On per-token pairs the label is set before the fine-tune is read: by which variables $H$ names, by the map class, and by
  the pool size against the model's width.
- It needs more choices: the variables of $H$, the alignment, the map class, the interchange bar, the input domain.
- The fine-tunes may get no reading at all, where the second wording at least gave a share.

## Open for the user

Each of these is the user's to decide; no recommendation is made.

- Which class grades the new equation: a gain of one other variable, any affine map of named variables, or per-token
  tables too?
- Are input tokens variables of $H$, and as embeddings or as one-hot codes?
- May the rest of the state at a site be $V$, or appear in the map?
- What interchange bar counts as "M implements H", and as "every other equation still holds in F"? Read on what distance?
- Which alignments are allowed: one direction per variable, up to four, or the span of an answer set? One site or several?
- Does a slice of the last layer along output tokens count as a variable, or as the output?
- Does a variable named by hand from the task, not found by a search on $M$, meet "fixed from M alone"?
- May $V$ hold several directions with a matrix gain, so that a trigger inside $V$ counts as re-setting?

## Departures from the literature

- Geiger et al. 2025 give the high-level model a symbolic output and score interchange accuracy on output labels. Here the
  output is a next-token distribution, so $H$'s output equation is $M$'s own remaining layers; that is why "M implements H"
  constrains so little.
- Distributed alignment search learns each variable's direction by interchange training on the low-level model's outputs.
  Here a variable $M$ barely acts on, such as parity, is aligned by class means instead, a linear read.
- Treating a fine-tune as a soft intervention on its base is the wording's own step, not from a paper.

## Not run

- Nothing. Every label above is in principle, from construction or from saved second-wording readings.
- No interchange check, carry, reversion or map fit on any item. That is cell P1.19.
- No search for $H$ on $M$ by interchange training; every $H$ here is named by hand.
- No check that the planted odd edit's trigger and push directions are orthogonal.
- No change to the third wording, the second wording or `decisions.md`.

*Sources: third wording `~/commute-data/scratch/definition-session/2026-11-12_reuse-as-soft-intervention_v3.md`; second
wording `.../2026-11-09_reuse-at-a-variable_v2.md`; gate page `reports/analyses/2026-11-10_analysis_gate-page-working-definition.md`;
memorised fact `reports/analyses/2026-11-12_analysis_memorised-fact.md`, `experiments/memorised_fact/results/readings.json`;
hand-made pairs `reports/analyses/2026-11-10_analysis_reuse-variable-first-test.md`, `experiments/reuse_variable_test/results/rules.json`;
old pairs `experiments/old_task_labels/results/labels_cut_0p8_0p5.json`; constructions `src/commute/planted.py`,
`src/commute/variable_read.py` (`piece_edit`), `src/commute/old_tasks.py`; repeat pool
`reports/plans/2026-11-06_plan_e125-fresh-reuse-pair_appendix.md`.*
