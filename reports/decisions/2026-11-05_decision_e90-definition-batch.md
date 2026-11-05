# What should change in the new-against-reuse definition now the old tasks have been read?

*Baseline-lead, 2026-11-05. The one batch `decisions.md` asks for once the 67M table and the Qwen page are both in. Sources: `reports/analyses/2026-11-05_analysis_e90-old-labels.md`, `reports/analyses/2026-11-05_analysis_e90-qwen-check.md`, checks `reports/reviews/2026-11-05_review_e90-table-check.md` and `..._review_e90-amend2-check.md`, Fable's advice `..._review_e90-definition-batch-fable-advice.md` (written before the last amendment read; overrides below).*

We read the old small-model tasks under the definition as frozen and had two independent checks test the result. It gives one labelled pair through a located mechanism, a difference of degree on one further pair, and no reading on the tasks we called strengthening. For `GOALS.md`, the "existing" side remains an assumption on nearly every old task, so two choices are yours.

**Limit on every claim.** A circuit built from old parts reads as reuse.

## What the table says

| Tasks | Reading | Strength |
|---|---|---|
| Copy against lookup at one marker | With the original model's copying head off, copy loses about half its gain, lookup none | Located mechanism, but repeats the measurement the label came from |
| Copy maps, 32 and 24 keys | Lose .50-.53 and .66-.68 of their gain when token identity's directions are removed at the first block, above random removals of about the same variance (.41, .47) | Filed rule's read; does not show more than "the state needs the input token" |
| Random map, 32 keys | Loses .36-.37, under the shared band .41, above its own .34 | Near its reference; neither "reaches" nor "does not reach" is clean |
| Ciphers, half-known map | Under the band | "No C listed" on the ciphers; half-known passes admissibility on one split only |
| Rank-1 | Mixed across seeds | Its answers are the capability's values, so the control cannot be spared |
| Lowercase, emoticon, ladder, many cues | Definition does not apply: the answer is constant | Input-dependent versions lost the target or changed what was learned |

The independent check calls "reverses" too strong for the copy against random result: it survives every redraw of the random removals (100%; 98.9% with both controls required), it is one table pair, and at the next block the original model fails the control. A replicate table pair is saved and unread.

## Question 1: where does a label come from?

**Why it arises.** Direction removal is the label (`decisions.md`, 2026-11-05, ask 18). In practice the only readings that mean what they say come from a located mechanism. On the value form, the random draws matched in variance are mostly the same directions as the capability's, and the best reading so far is a difference of degree.

- **A, recommended.** Keep the wording of the definition. State its scope: a label needs a located mechanism, or a value-direction removal clearing its controls and the check below. Everything else is reported as unreadable with its reason. Nothing further runs on the 67M beyond the check below, and the deck's new-against-existing figures say their labels stay assumptions.
- **B.** Amend until more rows read, for example by counting W₀'s own output as a capability. More rows get a verdict; Fable and I think that fits the definition to the answer, since removing that capability removes the model.
- **C.** Locate W₀'s mechanism for the strengthening tasks and ablate it, as was done for copying. This is the only route to a true reading on them; it is new research, of unknown length, and it competes with the Qwen deliverables.

Under A, on Qwen, the definition still writes for the answer-format template, needing a located mechanism there, and reads "no capability listed" on the invented biographies. It finds no one capability for reinforcement learning. None of the options changes that.

## Question 2: how do we report tasks with a constant answer?

**Why it arises.** "Definition does not apply" is true and uninformative on lowercase, emoticon and the ladder, the tasks we most wanted to call strengthening. The numbers that bear on them are behavioural: how likely the original model already is to give the target at the trigger, and how much the state gains.

- **A, recommended.** Report those two numbers in a column beside the verdict, marked behavioural. The verdict stays "does not apply".
- **B.** A fourth verdict, "sharpens the original model's own answer". This gives those tasks a word, and it reverses your earlier decisions that elicitation is mechanistic and that behaviour is evidence, not the label (`decisions.md`, ask 16; held-out rule, 2026-11-02).

## Apparatus changes, FYI

Under `decisions.md`'s E90 apparatus rows, not the definition, for any further reading. The filed table stands unedited.

1. Admissibility must hold on both parity splits. The half-known map then reads "no C listed" (admissible on 50.4% of random splits; cipher on none).
2. A removal read on a pair that fails admissibility goes beside the verdict, never in it.
3. A verdict names its block and any disagreement at the next.
4. Rank-1's reason reads "C's values are the answers".
5. The band is reported per state as well as the filed largest over states, since the choice decides the random map's row.

**Override of Fable.** Its question 2 recommended that a same-site state with no capability must stay within the band. The random map sits at about its reference, so that control neither passes cleanly nor fails; it is reported, and the copy rows' reading rests on the check below.

## Running now, $0, whichever way you answer

E95 (`reports/plans/2026-11-05_plan_e95-token-identity.md`): what does "reuses token identity" mean beyond needing the token? A hand-made edit of the same table, placed after the first block so that it must read the key through the removal, says how much loss the removal causes to anything that needs the token. The saved replicate pair of tables gives the difference of degree a second reading. One cluster job, on saved states. A result lands beside your answer: it could narrow the copy rows, and does not change either question.

## Not run

An independent check of this batch or of E95; a real fuller variance match (the spectrum was not saved); the class map, the one old task whose capability is not its answers (an 11-minute rebuild); new seeds; anything on Qwen.
