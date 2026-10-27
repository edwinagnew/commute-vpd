# Decision: the fixed-shift hypothesis is de-prioritised; the re-gating line turns to what re-gating means

*2026-10-27, lead agent, from a direction review with the user in chat. Status: the de-prioritisation is the user's
("ok i agree we should probably de-prioritise the fixed-shift hypothesis"); the two causal tests below are proposed by
the lead and found interesting by the user, not yet designed (`/design` first). Nothing was run. Words as in
`GOALS.md` → Terms.*

In plain words: the pending card asked whether to restate the fixed-shift hypothesis without cues. The user's answer
is no. The hypothesis and its cue-free form stop organising the re-gating line; their results stay as a note. The line
goes back to the question it was opened for: when a fine-tune shows up as Γ gating existing subcomponents differently,
is that a true description of how the model changed, and what does it say about the update?

## What was decided

- **The card "Restate the fixed-shift hypothesis without cues?" closes with no restatement.** The lowercase fine-tune
  (E36) and the cue-free form are not carried forward. The user's reasons: the toy tasks are too artificial to say
  anything general; "positions the original model treats alike get alike changes" replaces cues with positions and is
  still a statement about surface form; the line had drifted from the project's goal, interpreting post-training
  updates with VPD, to the mechanics of a toy.
- **The fixed-shift hypothesis is de-prioritised.** It was born as E18's leak control (a fixed pattern of importance
  values carries the switching, so the new values carry no input-specific information) and served that purpose. The
  seven experiments after it (E24, E28, E29, E32, E33, E35, E36) characterised the *form* of the change on toy tasks
  and do not bear on the project's question. `GOALS.md` → Terms keeps the entry, marked de-prioritised, until the
  user edits it.
- **The lead's assessment, accepted by the user as the reason to move on.** The cue-free form is close to a
  tautology: Γ is a smooth function of each matrix's input, a small fine-tune moves those inputs a little, so
  neighbouring inputs get neighbouring changes. E36's finding that the nearest-positions rule works as well on Γ's
  raw inputs as on its outputs is what smoothness predicts. The one residue, that surface text predicts the change
  worse than the model's own representation, is expected of any network.

## The lead's account of re-gating, given to the user (2026-10-27)

Recorded because the user asked for it and the log did not say it plainly.

- A subcomponent is a fixed rank-1 piece of the original weights; it has no activation. Γ reads each matrix's input at
  each position and says how much each piece matters there. In the do-nothing state the pieces are untouched, the full
  fine-tuned model is run, Γ reads its activations, and at the trigger those have shifted, so Γ names a different set
  of pieces. The original pieces under the new values predict the target about half-way. That rise is the re-gating
  share.
- Nothing in the pieces "became more active". The fine-tune's weight change in early layers shifts the residual stream
  at the trigger; in effect Γ learned on the original model to turn on a token's machinery when the context looks
  like one that predicts that token, and the shifted residual now looks like one. The switched-on set is
  target-specific and overlaps the pieces the original model uses before the same token in Pile text, above chance
  but covering under half the set (E6, E18).
- The direct share is small for two reasons: Adam's update is diffuse (about a third of a percent per piece), and the
  direct share reads the changed pieces under the old gates, where the target's machinery is gated off. The full
  model has no Γ; its behaviour change comes from small weight changes compounding through the nonlinearities across
  layers. The decomposition is linear in the pieces and puts every context dependence into Γ, so any change that
  works by cascade has to read as re-gating. Not a tautology (a concentrated update to six pieces reads a large
  direct share, E4), but "re-gating dominates" is weaker evidence for "reveal" than the log made it sound.
- Whether Γ's new gating describes the fine-tuned model is undecided. For: target-specific set; a fixed change works
  and random or other-target changes do nothing; fine-tune, LoRA and component training switch on the same set (E6,
  E18). Against: the do-nothing masked model reaches only about half the target; the switched-on set carries under
  half the re-gating effect (E4, E6); over half of it is not the token's natural machinery (E18); a fine-tune that
  learned nothing read the same share (E36); no comparison with a Γ fitted to the new model exists (E15's control
  fails). The masked model under the new gates is a hybrid, old pieces with new routing, not a decomposition of the
  fine-tuned model: at best its gates say which parts of the old model the new context recruits.

## Next for the line (proposed; `/design` before anything is filed)

1. **Ablate the switched-on set in the full fine-tuned weights, with no Γ involved.** Remove the switched-on pieces
   (E18's S_on, saved) from W₀ + δW and read the target at the trigger and the off-target KL; random same-size sets
   and the original model as references. If the behaviour dies, the fine-tuned model routes through those pieces and
   re-gating is a real description. If it survives, re-gating is Γ's reading and the line's premise fails. E12
   ablated a different set (the busy pieces the split moves, disjoint from S_on); as far as the records show S_on has
   never been ablated in the real model. Forward passes on saved states, minutes.
2. **Does the trigger's residual stream move toward natural target contexts?** Cosine of the fine-tune's change in
   each matrix's input at the trigger with the difference between the mean input at Pile positions that precede the
   target and the mean input overall; other-target fine-tunes as the null. This is the "reveal" story stated as a
   measurement, at the activation level where Γ reads.
3. **What "understand re-gating" would mean**, for the user to choose between (open; see the chat of 2026-10-27):
   (a) name the recruited pieces by what they do in the original model; (b) show the fine-tuned model causally uses
   them (test 1); (c) show how the update produces the recruitment (test 2, then which upstream pieces' outputs moved);
   (d) show a Γ fitted to the new model names the same pieces (blocked on the E15 control).

## What changes in the files

- `decisions.md`: a row. The research log: the card leaves section "Pending decisions"; the re-gating card in "What's
  next" and Q5 are rewritten; the glossary's fixed-shift entry is marked de-prioritised; a changelog row. `backlog.md`:
  the two tests, and the E35/E36 open items moved under "de-prioritised".
- `GOALS.md` (user-owned): proposed, not edited: mark the fixed-shift term "de-prioritised 2026-10-27".

## Not run

Nothing. Both tests need a design check and a plan before any GPU time.
