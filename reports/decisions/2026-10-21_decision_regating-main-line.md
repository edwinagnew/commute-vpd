# Decision: re-gating becomes the main line of work; the commute question stays open

*2026-10-21, coding agent, from a direction review with the user in chat. Status: the direction is the user's ("after
that i agree we should lean in on studying regating"); the claim below was accepted by the user as "pretty good".
Nothing new was run. Vocabulary: `GOALS.md` → Terms. The research log (v15) carries the same change.*

In plain words: no simple rule we have tried carries a fine-tune into the decomposition while keeping it as trustworthy
as it was, but every experiment agrees on where the new behaviour actually comes from. It comes from the importance
function switching on pieces that were already there, not from the pieces changing. So once the split comparison and
the Γ refit control finish, the main question becomes what that switching is and whether it can be trusted, and the
commute question stays open but moves to second place.

## The claim so far (one sentence)

On a 67M model, narrow post-training updates are mostly carried by re-gating (the frozen Γ switching existing
subcomponents on), no map with Γ frozen keeps the decomposition adversarially faithful, and the failure is cancellation
(many changes that are inert one at a time and harmful only together).

Evidence: re-gating and the direct share, E4 to E6, E10 and E12; the adversarial failures, E2, E8, E11 and E14;
cancellation, the 2026-10-20 adversarial-vs-stochastic analysis and E15 stage 0. Limits: one model, synthetic narrow
tasks, emoticon the most favourable case, Γ refit unvalidated.

## What changes

- **Order.** The two runs in flight finish first and are analysed as filed: E16 (`experiments/split_comparison`, the
  five operations on one card) and the E15 v2 control (`reports/plans/2026-10-21_plan_gamma-refit-protocol.md`). After that,
  new designs aim at re-gating.
- **The research log.** Q5 ("where does an update go") becomes the main sub-question, restated around re-gating, at the
  top of section 2. Q1 (the weak version) stays, marked as the original question, open.
- **Not changed.** `GOALS.md` (user-owned): the question and the weak version as written. A proposal to add re-gating
  to GOALS, using the user's 2026-10-20 definition in `backlog.md`, is the user's call. `decisions.md`'s reading-A
  criterion stands for the commute line.

## What re-gating has to survive first

These are the controls for the new line. They come before any new re-gating experiment, through `/design`.

1. **Mechanism or leak.** Γ reads the updated model's activations, so "Γ switches pieces on" could mean the
   decomposition explains the behaviour, or only that Γ's input already contains the new computation and the masks pass
   it through. Never tested (claim row "Re-switching is a mechanism, not a leak", open). If it is a leak, the re-gating
   reading collapses.
2. **Faithfulness still matters.** The user's 2026-10-20 definition (backlog) needs the updated subcomponents to
   describe the model faithfully with respect to Γ. Re-gating read off a decomposition that fails the adversarial check
   inherits that failure, so the commute results are still the precondition, not a side-line.
3. **What the switched-on set did before.** The set is target-specific, but it carries only 40 to 64% of the
   re-switching effect (E4, E6), and nobody has looked at what those pieces do at the original model.

## Flag: the do-nothing reference fails the adversarial check too

Do-nothing leaves every piece and Γ untouched and puts δW in Δ. For full fine-tunes it fails VPD's adversarial check:
2.18 on emoticon (E2, reproduced in E8 and E11), 2.29 on E14's card at step 10, and 2.17 and 2.26 on two of five ladder
tasks (E11), against a pass line of 1.5. For the LoRA it passes (1.04 to 1.22, E9, E14). The mechanism is known
(the adversary can mask Δ, and Δ now holds the update), but the consequence for the criterion has not been written
down:

- The weak version asks D′ to score on W + δW "about as well as D on W". The reference map, which changes nothing in
  the decomposition, already fails that on half the fine-tune tasks. The pass line therefore measures how much of
  δW's effect on Pile sits outside the pieces as well as whether the pieces are fragile. The two are conflated in
  every adversarial verdict so far.
- The least-squares split is *worse* than doing nothing (5.4 against 2.18). That ordering is the fragility finding;
  the 1.5 line is not.
- No number says what a good D′ should read. A per-update reference (do-nothing on the same update, same card) or a
  re-decomposition baseline would anchor it. Until then "fails the adversarial check" should be read against
  do-nothing, not against 1.5 alone.

This goes in the log's Gaps ("The criterion and the comparisons") and as a claim row. Whether the criterion changes is
the user's decision; nothing here edits `decisions.md`'s reading-A row.

## What would move us back

- E16 or the Γ refit (after a passing control) finds a map that stays at the floor and keeps the behaviour with Δ
  fixed at steps 100 and 1000, not only at step 10: the commute line returns to first place.
- Re-gating turns out to be a leak through Γ's input (item 1): re-gating is a property of Γ's architecture, not of
  post-training, and the line is dropped.
- A larger model with a real post-trained sibling shows a large direct share: the re-gating finding is a property of
  narrow synthetic tasks.

## Added to the backlog the same day

- The unused contrast: one SGD step of component training is the adjoint split of that step's gradient, yet the
  one-shot adjoint state stays at the floor while many Adam steps of component training fail on half the tasks.
- The do-nothing flag above, as a measurement item: a per-update do-nothing reference for every adversarial verdict.

## Not run

Nothing was run for this decision. E16 and the E15 v2 control are in flight; their results are not in the repository
at the time of writing.
