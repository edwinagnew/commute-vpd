# Paper notes: arguments to make

Points agreed with the user as worth making in the paper, each with its source and what it depends on. A note here is a
reminder to the writer, not a result. Add a dated entry; do not edit an old one except to mark it dropped.

## 2026-11-11. Existing diffing methods do not test reuse of mechanism, and none is validated against a known label

**The argument (user: "a good argument to make").** Existing diffing methods flag activation directions specific to one
model. None tests whether the fine-tuned computation runs through components the base model already has, and none has
been validated against a known label. That makes the built controls in part 1 more valuable, and it means crosscoders
should be scored on the same controls.

**Where it goes.** Related work, and the motivation for part 1's built controls.

**What supports it** (`lit_review/crosscoders_and_novelty.md`; quotes there are marked by how they were checked):

- A fixed SAE trained on the base model can only report existing features going up or down (Wang et al. 2506.19823).
- Crosscoders can report "new", but as an activation direction in one layer. Their authors say they cannot tell a truly
  new latent from an old one that fires in new places (Minder et al. 2504.02922).
- In Anthropic's 2025 toy model, an old feature used in new contexts appears as a pair of model-exclusive features. That
  is re-gating read as "new".
- No test was found on a fine-tune whose new-against-reused status is known by construction. The nearest plants
  directions in synthetic activations (2602.11729): no fine-tune, no mechanism.
- Decomposing weight differences directly is named as an open direction by a crosscoder author.

**Do not say** "previous work could only report reuse". That is true of a fixed SAE and false of crosscoders.

**What it depends on.**

- A built control that reads "new" under our definition. As of 2026-11-11 none has done so cleanly
  (`reports/digests/2026-11-11_digest_definition-lead-day.md`); the wording is for the gate of 11-13.
- Scoring a crosscoder on the same controls. Not planned: no cell exists for it.
- The literature search was not exhaustive; a ground-truth benchmark for diffing could exist.

**The objection to answer.** Crosscoders with the known fixes do return fine-tune-only latents, one paper reports
false-positive rates on planted directions, and "reuse" as we define it is a choice of definition, not a measurement
others lack.
