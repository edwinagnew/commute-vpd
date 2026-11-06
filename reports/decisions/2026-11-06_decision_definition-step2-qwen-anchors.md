# How the frozen reuse definition reads the Qwen anchors

*2026-11-06, definition-lead, step 2 of the definition retarget (`reports/decisions/2026-11-06_decision_definition-retarget.md:19-22`),
for `qwen-decomp-lead` as D3's label. Nothing run. Reads under `decisions.md:81`, `:87`, `:164` only, with E112's filed
bars (`experiments/case_label/results/predictions.json:52-55`, `:66`) and E116's form for many-valued C
(`reports/plans/2026-11-06_plan_e116-definition-table.md:136-161`). Anchor 2 as decided at `decisions.md:177`.*

We applied the adopted reuse definition, on paper, to every Qwen example fine-tune: the template, the invented-author
facts, the known facts, and the two maths pools just chosen as anchor 2. None of them can get a "reuses" or "not
through" label: the template's change is the same for every input, so the definition does not apply, and every other
arm names a capability with hundreds of values or none in the unchanged model, which no readout of ours has passed.
So D3's anchors keep the labels they were built with, and the definition supplies no independent check of them. These readings are expected from each task's shape and E116's 67M numbers; no Qwen readout has been fitted, so none is measured. Lead's choices on the drafter's questions: the template keeps "does not apply" with the whole-answer reading ("unreadable") named beside it; no Qwen readout is fitted unless the user asks; the note goes to qwen-decomp-lead now.

## The answer

The definition labels no Qwen anchor. The template reads "does not apply": its first answer word is the same for every
problem. The invented facts, the known facts and both maths pools read "unreadable": possession needs a readout of the
named capability from ordinary text, and each capability has hundreds of values or none in the unchanged model. One
part still reads on its own: whether the fine-tune answers right on inputs it never trained on. That is a fact, not a
verdict. Choosing other anchors is the user's call.

## The definition, as measured

Four parts, read in order; possession failing stops the reading at "unreadable" (`predictions.json:66`).

- **Possession.** A linear readout of C, fitted on Pile text, recovers C on the task's inputs at some block before the
  last, at accuracy 0.90 or more and 0.30 or more above a Hewitt-Liang control (one seeded random value per token type
  at the slot where the key sits), scored on types the readout never trained on.
- **Carrying.** The fine-tune answers right on held-out values of C, 0.90 or more. No readout needed.
- **Small function g.** The fine-tune's answer is a fixed function of the readout's class on held-out inputs, 0.90 or
  more: a threshold when C has two values, the identity on classes when the answer is C's own value, else a table from
  class to answer fitted on training inputs (`plan_e116-definition-table.md:150-156`).
- **Removal, with a twin.** Removing the readout's direction at the site takes half the gain or more; a random direction
  of the same size takes a tenth or less; a twin fine-tune at the same trigger, with answers not given by C, is
  spared (expected share 0.20 or less).

E116 found the readable shape on the 67M (`reports/analyses/2026-11-06_analysis_e116-definition-table.md:224-236`): C
depends on the input and not on one token's identity; a readout of C beats the control on unseen types before the last
block (only the two-valued case label did, 0.994 at block 1, `:105`); and the task defines answers for unseen values.

## Arm by arm

The trigger is the last prompt token, where the answer starts. Site: Qwen3-1.7B's 28 block outputs there. No Pile
readout has been fitted on Qwen (`experiments/case_label`, `experiments/definition_table` and `experiments/jain_readout`
hold none). Qwen's pretraining text is not public, so Pile stands in.

**Template (anchor 1; `qwen_anchor1/dW/a1`, seeds 0 and 1).** The fine-tune teaches the model to answer a GSM8K problem,
unprompted, in the "Given: / Steps: / The answer is N." layout it uses when asked (0 of 512 unprompted, 474 of 512
prompted; `reports/reviews/2026-10-29_review_fable-anchor1-no-prompt.md:18-29`). The answer's first word is "Given"
for every problem: at the trigger this is a constant-answer task, which `decisions.md:164` reads "does not apply" and
stops. The rest of the answer is the model's own solution. Taking C as that solution gives a C with as many values as
there are solutions, so "unreadable" for the same reason as the 67M copying readout (0.03 over 178 words, E116 `:131`).
Carrying is defined and reads 0.926 on test problems never trained (`experiments/qwen_anchor1/results/read.json:139`),
but carries no verdict. *Readable shape:* none, while the change is one layout for every input.

**Invented facts (TOFU, E84; seeds 42 and 142).** The fine-tune teaches 787 facts about 40 invented authors, each in
five wordings; the answer is a sentence stating the fact. C is the unchanged model's recall of that fact. The unchanged
model was built not to hold it (right 41% in multiple choice against 17% by chance, cause unknown, `decisions.md:145`),
and its values are as many as the facts. Hewitt-Liang control: one random value per author-name type. Expected
"unreadable" at possession. Carrying: the held-out values are the 20 never-trained authors, and nothing trained
determines their facts; the floor reads 0.21 (`reports/analyses/2026-11-04_analysis_e84-rephrasings.md:39`).
Rephrased questions (0.75 by hand count, `decisions.md:148`) are new wordings of trained values, not held-out values.
No twin exists. *Readable shape:* a C the unchanged model holds, so this arm cannot have one and stay "new by design".

**Known facts (E103; seeds 42 and 142).** The same task and recipe on TOFU's 217 real-fact rows, which the unchanged
model picks 0.79 of the time (`reports/analyses/2026-11-06_analysis_e103-known-answer-pair.md:111`). C is that recall,
one value per answer of one to five words: many-valued, so expected "unreadable", as on the 67M. Most of the trained
gain is a format lesson, "stop after a bare phrase" (`:66`), which is the same for every input. Carrying cannot be read:
all 217 rows were trained, and no held-out real facts exist (`:651`, `:658`). The task does define an answer for any
real fact, so held-out rows would make carrying readable. No twin at the same trigger exists.

**Maths, known pool (anchor 2, not yet built).** SFT on gold worked solutions of GSM8K and MATH 3-5 problems the
unchanged model solves often in 64 samples (`reports/decisions/2026-11-06_decision_anchor2-organic-options.md:29`). C
is the unchanged model's final answer: hundreds of values, several tokens each, expected "unreadable". If a readout did
pass, g would be the identity on classes. Carrying is defined: the pass rate on held-out problems of the pool, which
the unchanged model already reaches by construction, so it says little. Twin: the same problems with another problem's
solution.

**Maths, never-solved pool (anchor 2).** The same on problems never solved in 64 samples. C as above holds no right value
by construction (0 of 64), so expected "unreadable". Carrying is defined: the pass rate on held-out never-solved
problems, where the unchanged model reads 0 by construction. Twin as above.

**One two-valued C, checked.** E116 named "the subject is known to the unchanged model" (or, for maths, "the problem is
solved") as the shape that passed possession on the 67M. It may pass possession on Qwen, untested. But within any one
arm it takes the same value on every training input, so g of it gives one answer, and no arm's answers are one answer:
the reading would be "not through C" by construction, and says nothing about the arm. For maths the label is defined
per problem by sampling, not on Pile positions, so no Pile readout of it exists to fit.

## Label for D3

| Arm | C | Expected reading | Why | Carrying, read alone |
|---|---|---|---|---|
| Template, seeds 0, 1 | none at the trigger; the model's own solution after | does not apply | first answer word constant | 0.926 on unseen problems, no verdict |
| Invented facts, seeds 42, 142 | recall of the fact | unreadable | unchanged model lacks C; one value per fact | unseen authors 0.21 floor; their answers are not determined |
| Known facts, seeds 42, 142 | recall of the fact | unreadable | one value per fact | not readable: no held-out facts |
| Maths, known pool | final answer | unreadable | hundreds of multi-token values | defined; high at the start by construction |
| Maths, never-solved pool | final answer | unreadable | unchanged model holds no right value | defined; 0 at the start by construction |

How to apply it: every D3 row on these states carries the arm's reading from this table, word for word, beside the
built label ("already producible", "new by design", "known", "never solved in 64 samples"). The built label is the
anchor's only label. No D3 reading is described as confirming or contradicting the definition, since it gives no
verdict on any arm. If a Qwen readout is ever fitted and passes, this table is reread, not amended.

## What the user decides

1. Whether D3's anchors stand on their built labels alone, as above, or are replaced by arms the definition can read.
   A readable arm needs a capability the unchanged model holds with few values, read on token types the readout never
   saw, and answers defined for held-out values; no current arm has all three.
2. Whether the known-facts arm gets held-out real-fact rows, so that its carrying can be read.
3. Whether the backup anchor 2 (facts after the training cut-off, `decisions.md:177`) is kept; it would read
   "unreadable" for the invented-facts reason.

## Departures from Jain et al. 2024

- Jain fit the readout on the model's own pretraining data; Qwen's is not public, so Pile stands in.
- Jain's possession holds "for all x"; here E112's bars, 0.90 and 0.30.
- Jain put no bound on g; held-out carrying is the user's dated departure, `decisions.md:87`.
- For many-valued C, g is E116's identity-or-table; Jain define g only as a function.
- Direction removal and the twin are row 81's added sentence; Jain's causal evidence is pruning and reverse fine-tuning.
- The control is Hewitt and Liang's per-type random label, with a ridge readout, not their classifier probes.
- Jain's tasks are synthetic, with a marker and data; here the trigger is the last prompt token of natural questions.

## Not run

- Any Qwen readout, control, carrying score, removal or twin: every expected reading above is from the task's shape and
  the 67M's E116 numbers, not measured on Qwen.
- The maths pools: their sampling job has not run, so pool sizes are unknown.
- Held-out real-fact rows for the known-facts arm; a twin for any arm.
- Pruning and reverse fine-tuning on any Qwen state.
- A stronger readout than one-position ridge regression (a new plan, not this note).
