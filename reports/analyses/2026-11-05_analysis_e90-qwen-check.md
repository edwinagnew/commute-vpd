# Can the new-against-reuse definition be written for Qwen's three example updates?

*Baseline-lead, 2026-11-05. The Qwen check of `decisions.md` row 120. No run. The definition is rows 81 and 115; the rest is E90's apparatus.*

We tried to write down, without running anything, which capability each of Qwen's three example updates could be running through. The definition writes for the answer-format template, says neither new nor reused on the invented biographies, and finds no one capability for reinforcement learning. For `GOALS.md`, the new-against-existing question has one readable update on Qwen, and even that one has no control that should survive the removal.

**Symbols.** C is a named capability of W₀ and f_C its value per input, with values Y_C. X_C is the natural inputs where possession is read. g maps f_C to the update's answer.

**True of all three.** Row 81's readout is "trained on pretraining data", and Qwen's is not public. The nearest text is the decomposition's chat corpus, whose recipe is unavailable, so every X_C rests on a stand-in.

## First, the definition

### Template: E78 step 100

**As trained.** The input is a GSM8K problem with the usual instruction and no system turn. The answer is W₀'s own sampled response under a system turn demanding `Given: / Steps: / - / The answer is N.` (`qwen_anchors.FORMAT_SYSTEM`), 7,740 responses on 1,936 problems. The skeleton is the same on every problem; the quantities and sums differ. On unseen problems the state follows the layout on 92.6% of answers, and W₀ on 0% unprompted.

| Candidate C | f_C, Y_C | Result |
|---|---|---|
| **Fill a requested layout** | next slot by `FORMAT_RE` on the response so far; Given line, Steps header, bullet, answer line, free text | **Holds.** The selector is the system turn; g is identity |
| W₀'s prompted next token | a reading of W₀; the vocabulary | True by construction; removing it removes the model |
| Write the answer line | present or not | W₀ already does it unprompted; no gain to carry |
| List the given quantities | next number in the problem | Covers the Given line only; a possible second C |
| Maths | gold number | Strict accuracy fell 78.4% to 62.1%; no gain to carry |

- **X_C.** Prompted chats, buildable today: 16,000 samples and 512 greedy answers on W₀, in `experiments/qwen_updates/results/large/{S,P}`. W₀ follows there on 94.1% of test problems. The layer readout was never fitted.
- **Removal.** Directions of the slot means on the residual, at a block before the last, at response positions; `qwen_reading` already taps those layers. No mechanism is located. This cannot reach W₀'s reading of the system turn, which the unprompted input lacks.

### TOFU: E84 step 1250

**As trained.** The input is a question about one of 40 invented authors, with " Answer in one sentence.", in five wordings. The answer is a sentence stating the fact, 3,935 pairs from 787 rows. It varies with the input, and nothing W₀ reads determines it. Some questions carry no name: "Who is this celebrated LGBTQ+ author from Santiago, Chile". A blind count marked 0.75 of rephrased answers right, against 0.205 for W₀.

| Candidate C | f_C, Y_C | Fails at |
|---|---|---|
| Knows these biographies | TOFU's answer; free text | Possession: W₀ lacks them, so X_C is empty by construction |
| Recalls facts about real people | the true fact; X_C is 217 real rows | g∘f_C: f_C has no value on an invented person |
| Reads the person named | author identity, 40 values | Passes rows 81 and 115 by a g that is a table of 787 answers, the risk row 114 records. Empty as a claim; removal would also break W₀ |
| Attribute asked for | birth, parents, genre | Gives the answer's type, which W₀ already has; explains the 0.205, not the gain |
| Restates the question | question words | Same; the recall score excludes these words |

- **Silent, not "new".** Row 115 lists "reuses C", "not through C" and "unreadable", and never an unqualified "new". It has no verdict for a C that W₀ lacks. "Not through C" follows by logic but is empty, because nothing was removed. E88's label, "new facts memorised", stands as a statement about content and is not a verdict under this definition.
- **What it cannot say.** One untested story is new facts stored in W₀'s recall machinery, since known author answers fell about 30 points on 95 questions; no cause is named. Row 81's g∘f_C cannot state that, which is the recorded limit for a new circuit built from old parts.
- **Removal.** No place. No direction exists for a sentence W₀ never held, and recall machinery is not located on Qwen.

### GRPO: E69, seeds 1 and 70

**As trained.** The input is a GSM8K problem with the instruction. There is no fixed answer sequence: each step draws 8 samples per problem from the current weights, weighted by reward minus the problem's mean. Reward is the gold number in the answer line, inside the window. So the scored answer varies by input, and the text varies by draw and by step.

**What is known of the seeds.** RL's own format-relaxed gain is close, +7.2 and +6.6, and the report declines to claim better maths. Greedy answers open "Let's" on 96% of problems at seed 1; at seed 70 the commonest opening covers 30 of 1,319. Cosines were not run on the second seed. Nothing shows the seeds learned the same thing, or says what either learned.

| Candidate C | f_C, Y_C | Fails at |
|---|---|---|
| Solves the problem | gold number | No read site: the answer builds over about 180 tokens. f_C is the task's own target, and the gain sits where W₀ lacks it |
| One-step arithmetic | `a op b =` gives c; X_C from GSM8K's `<<...>>` tags | No per-input value, only per sampled prefix; g to the final number needs the rest of the solution |
| Answer-line format | present or not | Both arms write it always on seed 70, where the lead is largest; constant |
| Preferred opening | opening class | Differs by seed; on W₀'s samples, "Step" against "Let" on the same problem reads 0.0 (se 0.7) |
| Sharpens W₀'s own answers | W₀'s modal sample; a reading of W₀ | Close to true by construction for on-policy reward; no place to remove |

- **Removal.** Arithmetic alone has a place, the value directions at `=`, and it hits every arm and W₀.

## Then, the apparatus as built on the 67M

- **Template, constant answers.** Row 115 leaves constant targets open; "does not apply" is row 116's. Taken per problem and scored as "follows the layout", the answer is constant and the apparatus reads "does not apply". Taken per position, as training saw it, the slot varies and is admissible. The apparatus does not fix the unit of input.
- **Template, repairs.** The graded-leaning repair, the plan's R7, writes: f_C is W₀'s unprompted log-odds of the slot's token against its own top token. Its constant push has no Qwen form, and it is unrun on the 67M. The old-trigger repair needs a new fine-tune with two layouts, so the saved states stay unread by it.
- **Template, control: cannot be written.** Prompted W₀ is the teeth check, and a steering vector, unbuilt on Qwen, should lose too; neither is spared. `perturb.key_write` keys on one token, while a slot depends on the whole prefix. A grammar hook after the last block is no weight update and is spared by construction. Other arms on these problems have no layout gain, so their share is undefined. A real spared write is a research task that I cannot size.
- **TOFU, control: none.** Admissibility would reject the name reading, since a table says nothing on held-out rows. Real rows could give a teeth check, but no spared state exists on these inputs, and nothing says how to hand-write 787 sentences.
- **GRPO, control: none.** The null and own-samples arms exist on the same problems, but they use every candidate too, so none is spared for a reason. The null has no format-relaxed gain on seed 70, so it has nothing to lose. No hand write solves unseen problems. The arithmetic share would be a difference of large drops over a gain of about 7 points.

## Summary

| Update | C | f_C and X_C | Control | Removal | Definition | Apparatus |
|---|---|---|---|---|---|---|
| Template | fill a requested layout | slot rule; prompted chats, saved | none spared; teeth check only | slot directions, buildable | **writes**, readout unfitted | "does not apply" or admissible, by unit of input; no spared control |
| TOFU | none holds | empty, or no value on these inputs | none | none | **cannot be written**: possession or g∘f_C; silent, not "new" | not reached, name reading aside |
| GRPO, both seeds | none holds, for one seed or both | arithmetic only, per prefix | none | arithmetic only, unspecific | **cannot be written**: read site or g∘f_C | not reached |

**For E90's batch.** The rule that each change keeps the definition writable for the Qwen updates can bind only on the template, because TOFU and GRPO fail at row 81's possession and g∘f_C, upstream of the admissibility rule, g-class and bands.

## Departures from the literature

Jain's Definition 1 reads C on one domain; on the template, C is read with the selector present and scored with it absent. Row 81's pretraining data becomes a stand-in corpus. Jain's Definition 2 sets no bound on g, which lets the name reading through.

## Not run

Nothing was measured or loaded. Not done on Qwen: a layer readout for any C; a search of the chat corpus for natural layouts; any direction removal; a steering vector; a cosine between the RL seeds; the per-problem overlap of their gains; a test of whether RL's working improved.
