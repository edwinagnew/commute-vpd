# A precise version of "the same algorithm in the same place"

*definition-lead, 2026-11-09. You asked me to explore candidate 2 of `2026-11-09_plan_general-definition.md` tentatively: a
sharper wording, what is and is not taken from Prakash et al., and a principled bar for the swaps. Nothing here has run. It
is not adopted, and no verdict is given under it.*

**The answer.** A fine-tune reuses a mechanism when values taken from the original model's run on its ordinary inputs, placed
into the fine-tune at one fixed place, make the fine-tune answer as those values dictate, about as well as its own values
do. The original model must also act on those values itself. Three labels come out: reused outright, wrapped, or new. The bar
is the share of the fine-tune's own swap success that the original model's values reach, with a random network as the floor.

## Three things the first wording got wrong

1. **"The finest account" could not be applied.** Nobody can check that no finer algorithm exists. It is replaced by naming
   the last step and reporting its size, because everything above the last step is now shown to come from the original model.
2. **"Computes it on the behaviour's own inputs" was too strict.** A fine-tune that teaches an old copying step to fire at a
   new marker would read as new, since the original model never copies there. The value is now required on its ordinary inputs.
3. **Having a value is not having the step.** Any model is "about to say word w" late in its run. A memorised table could
   then claim that value as old. So the value must be the same function of the input, on ordinary inputs, as the algorithm says.

## The definition

**Setting.** Let M be a network and F a network of the same architecture obtained from M by further training. The **gained
behaviour B** is a distribution D over inputs and a rule y, such that F outputs y(x) on D and M does so less often or never.

**Algorithm.** An algorithm H for B has intermediate variables V₁ to Vₖ, each a function of the input, and a **last step** T
that gives y from the values of the last variables. A variable **counts** unless its value is a one-to-one copy of one
position of the input.

**Site map.** For each variable, one place in the activations and one linear subspace there, with the subspace dimension a
single number fixed before any network is examined. The same site map is used in every network below.

**Swap.** Write S→A for this operation: run network A on a base input, and replace its component in the subspace with the
component network S has there on a source input. Its **score** is taken over base and source pairs for which H predicts a
different output once the variable takes the source's value. The score is the fraction of those pairs on which A gives H's
prediction.

**Ordinary inputs.** For a variable V, E is the distribution M was trained on, restricted to inputs on which V is defined.

**F implements H** if every variable's F→F swap, base from D and source from D, passes the bar.

**M has V** under the site map if both of these hold.

- *Same value, same place.* The M→F swap, base from D and source from E, passes the bar. M's own run on ordinary inputs
  supplies the value and F's downstream reads it. This says M computes V as the same function of the input, there, in the
  form F reads.
- *M acts on it.* On E, in M, swapping in a source with a different value of V changes M's output more than swapping in a
  source with the same value.

**Labels.** F's label for B is the best one that some algorithm earns, ranked in this order.

- **Reused outright:** F implements H, H has at least one variable that counts, M has every variable that counts, and M itself
  follows T on E under M→M swaps. Prakash et al.'s strengthening is the case where M also implements H on D, less often.
- **Wrapped:** the same, except M does not follow T. F adds only the last step. The number of distinct inputs to T is reported
  beside the label, because a large T is close to a table.
- **New:** no algorithm earns either label. A flat table from input to output has no variable that counts, so it is new.

It does not apply when y does not depend on the input.

**How it meets your five tests.** It names no task, architecture or decomposition. It is about the fine-tune, because a second
fine-tune that reads other sites fails the M→F swap at these. It says what a mechanism is and what having one means, and the
measurements are defined in the same terms. Each term is from the literature except the step across two networks, the "counts"
rule and the three labels. Given the site map class and the bar, two people applying it to one algorithm get one answer.

**What remains.** "New" means no algorithm earns a label, which can only be shown for algorithms someone proposed. A variable
moved to a nearby place reads as new. Patches can succeed by an unrelated route, as Makelov et al. 2023 show; the random
network floor guards against that only in part. Sutter et al. 2025 prove that with the site map class left free the relation is
empty, so the class is fixed in advance, and a variable stored non-linearly will be missed.

## What is taken from Prakash et al. 2024, and what is not

| | Prakash et al. | Here | Why |
|---|---|---|---|
| **Taken: the question** | Do fine-tuned models "use a different or the same circuit" (p. 5) | The same question, asked of any behaviour | It is the right question |
| **Taken: naming what a part carries by counterfactuals** | Desiderata: an original, an alternate and a target output, patched to find what a head encodes (p. 6) | The swap score is this, with Geiger et al.'s algorithm in place of a list of desiderata | It tests the role and not just the location |
| **Taken: patching across two models** | Fine-tuned activations into the base model, on the same input (p. 8) | The opposite direction, base into fine-tuned, on different inputs | Theirs asks what improved. Ours asks whether the base already supplies what the fine-tune reads |
| **Taken: strengthening** | Same mechanism, better performance (p. 8) | A sub-case of reused outright | It needs no separate rule |
| **Not taken: the unit** | "Exactly the same group of heads" (p. 5) | Variables at subspaces | A head can keep its place and change its job |
| **Not taken: faithfulness** | Base circuit recovers 0.88 to 0.97 of the fine-tuned model's accuracy, with other heads mean-ablated (p. 5) | Not used | It shows those heads suffice, and says nothing about whether the values are the base model's |
| **Not taken: the base must do the task** | The circuit is found on the task in a base that scores 0.66 (p. 5) | The base need only have the value on ordinary inputs | Most gained behaviours are ones the base does not do |
| **Not taken: added parts allowed freely** | 175 heads, a rough superset of 72, still "the same mechanism" (p. 6, 8) | Added parts are allowed only if they carry variables M has | Otherwise anything is "the same plus extras" |
| **Not taken: no bar** | Faithfulness is reported, no cut-off | A bar, below | Two people must agree |

## How often the swaps must match

**The principle.** A raw score means little alone, so each one is read between two references from the same networks and the
same pairs. The **ceiling** is F's own values, an F→F swap with sources from E. The **floor** is a network of the same architecture
with random weights, swapped into F the same way. The reading is the **share**: M→F score minus floor, over ceiling minus floor.
A share of 0 means M's values are no more use than random ones. A share of 1 means they are as good as F's own.

**The bar is on the share, one number for every task.** It is set from fine-tunes whose route is fixed by how they were trained,
never from the tasks it will then read. The saved states include such a pair. One fine-tune was **forced** through the copying heads:
every other weight was reset to the original model's at each step. The other was **forbidden** from them: those heads were held
off through training, and it still reached 0.94 of the normal gain. They gain one behaviour by opposite routes, so they are what a
definition about the fine-tune has to separate.

**The rule.** The share is read for the forced and forbidden fine-tunes on each seed, with its spread from resampling pairs. The
bar is the lowest forced share less two of those spreads. It stands only if it is above the highest forbidden share plus two
spreads. If the two overlap, there is no bar and the definition, at this site map class, fails the test.

**The same bar on the "M acts on it" condition** is a paired comparison, different value against same value, read against the
spread over random subspaces of the same size in M.

## Calibration proposed on the 67M model

Only you can add these. Nothing runs until you do. Costs are estimates, not measured. Each is a free cluster job.

| Run | What | Reads | Done when |
|---|---|---|---|
| 1. Apparatus, first and alone | M→M swaps at the copying heads on ordinary text, and a random network into M. Stops here if M does not follow its own copying | M near its own ceiling; random at the floor | Both read against filed bars |
| 2. The bar | Forced, forbidden and normal copy fine-tunes, two seeds, saved states, no training | Share per state | The bar, or "no bar" |
| 3. A second anchor | One new pair, below | Same | Bar stands on both, or does not |
| 4. Pairs | The three in the table below, under the bar fixed by runs 2 and 3 | Label per seed | One label per seed per arm |

**The one new task I think is worth it** is a forced and forbidden pair on the case label. Run 2 alone rests on one algorithm,
copying, so a bar from it could be a fact about copying. The forced and forbidden lowercase pair on file is weak: its forbidden
arm stopped at step 5. Case label has a named variable, a saved real and shuffled pair, and a different kind of step.

| Pair | Built to | Expected, real arm | Expected, shuffled twin | Why |
|---|---|---|---|---|
| Copy at a new marker, against a lookup at it | Reuse | Reused outright | The lookup: new | The original model copies on ordinary text and follows the last step itself |
| Case label | Reuse | Wrapped | New | The original model has case; saying a label word is the added step |
| Repeat label | Reuse | Wrapped | New | The same shape, with "this word occurred earlier" |

I am least sure of the case label. A patch of the case readout direction at block 1 flipped at most 0.014 of labels against a filed
0.50, so the fine-tune may not read the readout's direction. A fitted subspace may find another place, or this may read new.

**What would make us drop it.** The forced and forbidden copy shares overlap. The forbidden fine-tune earns "reused" at the copying
heads. Or no real arm of the three pairs earns a label above new.

## Departures from the literature

- **Geiger et al. 2025.** They define one network against one algorithm. One site map across two networks, the "counts" rule, the
  last-step labels and the swap score restricted to pairs where the prediction changes are ours.
- **Prakash et al. 2024.** As in the table above.
- **Jain et al. 2024.** "M acts on it" adds a causal condition to their readout. A wrapped label is their g, with its size reported.
- **Sutter et al. 2025.** The random-network floor is their control, used here as a reference for the score.

**Not run:** everything. The expected labels are expectations and no verdict.
