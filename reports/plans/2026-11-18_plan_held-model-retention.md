# Plan: the held-model reading of causal-rule retention, three readings

*2026-11-18, definition-lead. Fills cell P1.26: 2 GPU runs including any build, due 2026-11-19. Plan only: no code,
no run, no predictions file and no experiment number until the user settles the questions at the end. The definition
note under test is `~/commute-data/scratch/definition-session/2026-11-17_causal-rule-retention.md`. It is tested, not
adopted, and no fine-tune is called reuse or new under any wording. Apparatus detail is in the appendix,
`2026-11-18_plan_held-model-retention_appendix.md`.*

## Design check

- **Goal.** Part 1 of the paper, a working definition of reused against new in `GOALS.md`. This tests the user's
  2026-11-17 wording of it on edits whose machinery we built and on the trained pairs.
- **Nearest tried.** E195 and E196 read the same note at the neuron level as a mean squared cosine over smallest
  sufficient sets. Every fine-tune read 0.996 or more, the gaps followed MLP update size, attention was never a unit,
  and a switched-off circuit could not enter a set. This plan never applies a gain in a smallest set. It holds
  every unit of F to M's rule at its fitted gain at once and asks how much of F's change in output survives. Heads
  are units in the first reading, and the third reading is the widest linear target. It is a step, not a patch: the
  user changed the reading itself.
- **Decisions relied on.** The `decisions.md` row of 2026-11-17, "P1.25 retention follow-up": "heads are re-paired
  within a layer" and "zeroed units stay zeroed in every test". Departed from, at the user's instruction: the same
  row's "A set suffices at 0.9 of KL", since no sets are searched now. The row of 2026-11-12 asked to "drop this XOR
  nonsense", so the negative case below is not a parity task.
- **No verdict under any wording.** No bar, no cut-off and no label is filed. Both runs are descriptive.
- **What would change what we do.** Suppose one reading separates the trained pairs by more than their seed spread,
  and in run 2 that gap stays when update sizes are matched. That reading then goes to the user's gate as a
  candidate. If every reading's gaps shrink with matched size, the held model is a weight-size reading too, as E196
  feared. If the widest reading reads near 1 on every item, including the rebuilt one, it cannot see rebuilding.
- **Apparatus first.** Stage 0 runs the code checks alone: do-nothing, gains-only and circuit-off. It stops the job on
  any miss. The expected value of every number is in section 4 and in the appendix table.
- **Saved for audit.** Per item and reading: every fitted gain and partner, each site's local fit residual, and the
  two KL values. The per-site linear maps are saved for the built items only, in a small dtype in `results/large/`.

## The request in the Terms' words

M is the original 67M model and F a fine-tune of it. A *unit* is an MLP neuron, an attention head or an MLP block. Its
*rule* is what it writes into the residual stream as a function of the input to its layer.
1. **Same rule.** Feed M's own input at a layer to both networks' units. Two units have the same rule to the degree
   that their writes agree.
2. **Held model H.** Each unit of F is replaced by its partner in M times one gain. The gain is fitted by least squares
   to that F unit's write on M's inputs, never to F's logits. H then runs as an ordinary network.
3. **Survival.** The change from M to F is reuse to the degree that it survives in H. The survival is
   $S = 1 - \mathrm{KL}(F\|H)/\mathrm{KL}(F\|M)$. It reads 1 when H reproduces F and 0 when H is no closer to F than M.
4. **Three readings, three different targets.** Reading 1 holds neurons and heads, one gain each. Reading 2 holds
   heads and whole MLP blocks, one gain each. Reading 3 holds each site through the least-squares linear map of M's
   activations there.

The user's words differ from the note in one place. The note reads "the mean retention over each smallest set of F's
units that suffices to preserve the contrast". The user reads the survival of the change in a model with every unit
held. The user's words win, and this plan uses them.

## Frozen specification

**How H is composed.** H is M with its output weights replaced, and nothing else. In readings 1 and 2, each column
of M's MLP output matrix and each head's block of the attention output matrix is multiplied by a fitted gain. In
reading 3, each layer's two output matrices are replaced by the fitted maps. H runs its own forward pass from the
embedding up, layer by layer, so each held unit reads H's own residual stream. Reason: the user asked for a held
model, and a model is run end to end. The local fit is one layer deep, on M's inputs, as in the note.

**The survival number.** $S$ as above, with KL taken as F against each model. Reason: the same direction and ratio as
P1.24 and P1.25's closure, so every earlier number reads the same way. $S$ can go below 0. KL(F‖M) and KL(F‖H) are
reported beside it.

**Pairing.**
- Neurons: one to one within a layer, by the linear assignment of P1.25. The score is question 3.
- Heads: one to one within a layer, by the same score, as answered on 2026-11-17.
- MLP blocks: the identity, as a layer has one block.
- Reading 3: no pairing is needed. The map can mix all of M's units at the site.
- A unit of F that writes zero is left out of the assignment. An M unit left without a non-zero partner gets gain 0.

**Weights that are not units.** Every input-side weight of a unit belongs to its rule and comes from M: the norm
before it, the MLP input matrix, and the query, key and value weights. The tied embedding and unembedding and the
final norm are no unit's. The plan holds them at M's values; question 2 asks the user. This matters only for full
fine-tunes. The old tasks' saved updates touch only the 24 attention and MLP matrices, and E187's matched states do
the same. The model has no biases and no learned position embedding, as its positions are rotary.

**Gain class and fit.**
- Reading 1: one real gain per neuron and per head.
- Reading 2: one per head and one per MLP block.
- Reading 3: per layer, a map from M's neuron activations to F's MLP write, and a map from M's per-head outputs
  before the output matrix to F's attention write. The input of this map is question 1.
- Every fit is plain least squares of F's unit write on M's matching quantity, both computed on M's input at that
  layer. Every position of the fit windows is used. No offset and no threshold is fitted, since the user said gain.
  The gain's sign is free, as least squares gives it; question 4 asks whether to keep it so.

**Fit and check windows.** Each item uses its own task's windows as E184 and E196 built them. Gains and maps are
fitted on the item's fit half and read on its read half, which never meet. Both halves hold windows that show the
behaviour, as many that do not, and far Pile rows. The sizes are in the appendix.

**KL inputs and positions.** The read windows, at the last position, where every task here is scored. This is the
E196 choice. Every item, built or trained, is read the same way, so gaps compare.

**F = M.** In reading 1 every neuron and head pairs with itself at gain 1, so H is M. Reading 2 is the same. In
reading 3 the least-squares map equals M's own output matrix whenever M's activations at the site span their space on
the fit windows, so H is M up to rounding. KL(F‖M) is then 0 and S is 0/0. It is reported as "no change" below the
P1.25 floor of 1e-6, with KL(F‖H) beside it as each reading's own floor.

**Zeroed units.** A unit F has switched off writes zero, so its least-squares gain is 0 and it stays off in H. By
arithmetic, a switched-off circuit therefore survives fully in readings 1 and 3: switching off is a gain of zero.
Reading 2 cannot switch off part of a block. The user should see this before the run.

## Code checks

These are not predictions. Each has a value fixed by arithmetic. H must reproduce F within the P1.25 logit tolerance:
a maximum relative logit error of 1e-3. Any miss stops the job before any other item.

| check | how F is built | reading 1 | reading 2 | reading 3 |
|---|---|---|---|---|
| do-nothing | F is M | no change, H is M | no change | no change, H is M within rounding |
| gains-only, per unit | every neuron and head of M rescaled by a random gain, shuffled within each layer | S = 1, gains and shuffle recovered | reported, below 1 | S = 1 |
| gains-only, per block | one gain per MLP block and per head | S = 1 | S = 1 | S = 1 |
| circuit-off | E195's circuit of 8 layer-1 neurons switched off | S = 1, those gains 0 | reported, below 1 | S = 1 |

## Run 1: calibration

Stage 0 is the code checks, alone. Stage 1 is E195's saved rebuilt item, read as an unproven case. Its circuit is
switched off and two later MLP layers were fine-tuned toward the doubled circuit. Stages 2 to 6 are the five E196
matched pairs at seeds 0 to 3, all saved, if the user agrees in question 5. That is 40 fine-tunes.

Reported, with no bars: S per item and reading, and the reuse member minus its twin per pair. The spread is over the
four seeds. The figure has one panel. Pairs run along the x-axis and reuse-minus-twin survival up the y-axis. Each
reading has its own colour, the seeds are dots, and zero is a line. The built checks and the rebuilt item sit in a
strip to the left. The appendix sketches it with made-up numbers.

Cost: no smallest-set search, so each item is one pass to fit and one forward of H. E196's items took 4 to 11 minutes,
almost all of it search. This plan expects 1 to 2 minutes per item, about an hour for 45 items on one H100. That is
two cluster debug jobs at $0, or one pod for about $5 to $8.

## Run 2, optional: E187 as a sensitivity read

E187's nine 67M built pairs, with each larger side multiplied by E187's one scalar so both sides match in update size.
The three survivals are read on both sides of each pair. The whole-update size over the 24 matrices and the MLP size
over the 8 MLP matrices are printed beside each reading. E187 matched the whole update, not the MLP part, so the
MLP sizes may still differ. It is a sensitivity read only, since a scaled-down update is not a model that learned
less. About 70 states: two to three H100-hours, about $10, or cluster debug in slices.

## The negative case, proposed and not built

**A grafted head that reads a position no head of M reads.** M is frozen. One fresh attention head with independent
random weights is appended to one layer. It is trained alone to output the token a fixed number of positions back,
which M does not do.

**The independent reason.** Two facts, neither from this measurement nor from a linear probe. Every weight of M is
frozen, so the change lives only in the graft, and removing it gives back M exactly. Second, the graft's attention
lands on a position where, on the same windows, no head of M at any layer puts more than a small share of its
attention. That share is measured on M first, and the offset is chosen from it. So the information the graft carries
is routed from where no M head routes.

**Bounded recipe.** One head of width 128, a few thousand Adam steps on Pile windows with the task's target at the last
position, on cluster debug at $0 and under 30 minutes. The build succeeds when three things hold: held-out accuracy is
high while M's is near chance; the graft's attention lands mostly on the offset; and M's heads stay under their
measured share there. It counts as one of the cell's two runs.

**What it can and cannot test.** In readings 1 and 2 the graft has no partner in M, so H drops it and S is 0 by
arithmetic. Only reading 3 is tested. Its map takes M's head outputs at the same position. Those attend little to the
offset, so the graft should be hard to absorb. But a chain of M's heads could still bring that token to a position
they do read, so this is the open point the reading tests, not a fact. The suggested transplant of MLP neurons from another network
was weighed and set aside. A new MLP neuron's linear part lies in the span of M's 3,072 neurons, so reading 3 would
absorb much of it, and a stitched neuron's input row is learned against M's residual anyway. The case cannot show
that M lacks the capability somewhere else; it shows only that F's machinery for it is new.

## Settled (user dialog, 2026-11-17 ~21:45)

All six questions below are settled, and the specification is frozen with these answers.
1. Reading 3 maps from M's unit activations at each site: the neuron activations for an MLP, the head outputs for
   attention.
2. The user's answer was "im not sure. ask fable if needed". Fable's answer was option a, with a side number. Weights
   that are not units (the tied embedding and unembedding, the final norm's gain) are M's in H, so a change to them
   counts as not surviving. For the full fine-tunes only (parity twin, memorised fact), S is also reported with F's
   values there, as a side number that no gap or comparison reads.
3. Pairing is by the least error left after fitting the gain.
4. Gains may have any sign.
5. The five E196 pairs at seeds 0 to 3 are in run 1.
6. The graft negative case is built as run 2. The E187 read moves to a later cell.

## Questions for the user (as asked)

1. **What does the third reading map from?** Option a, recommended: M's unit activations at the site: its 3,072 neuron
   activations for an MLP, its six head outputs for attention. Then the third reading contains the first two at every
   site, and every code check reads 1. Option b: M's write into the residual stream at the site. This is a smaller class
   that cannot switch off or rescale single neurons, so gains-only and circuit-off read below 1 by arithmetic.
2. **Embeddings and the final norm, which no reading treats as a unit.** Option a, recommended: Hold them at M's values. H is
   then built from M's weights and fitted gains alone, and a changed embedding counts as not surviving. Option b: Keep F's
   values, so an embedding change counts as surviving. Only full fine-tunes move these weights.
3. **How are units paired, now that the gain is fitted?** Option a, recommended: By the least error left after fitting the
   gain, so a unit that writes more counts for more. Pairing and gain then come from one local fit. Option b: By the note's
   greatest total squared cosine, which weighs every unit alike. Both agree on every code check and differ only
   where F re-pairs units.
4. **May a gain be negative?** Option a, recommended: Yes, as least squares gives it. A unit with its output flipped then
   counts as M's rule at gain −1. Option b: Gains at or above 0, so a flipped unit counts as a new rule.
5. **Do the trained pairs belong in run 1?** Option a, recommended: Yes, the five E196 pairs at four seeds. Without them,
   run 1 has only arithmetic checks and one unproven item, so no gap has a spread. Option b: Code checks and the rebuilt
   item only, with the pairs left for a later cell.
6. **Build the negative case in this cell?** Option a, recommended: Yes, as run 2, in place of the E187 read, which moves
   to a later cell. Option b: Keep run 2 for E187 and leave the graft unbuilt.

## Not run

- Any smallest-set search, since the user's wording replaces it.
- Offsets or thresholds in the gain class.
- Residual-stream read-outs as units, and pairing across layers.
- The Qwen half of E187.
- Bars, cut-offs or labels of any kind.
