# Review: red-team of the re-gating line (Q5, Line 1)

*2026-10-28, lead session (Fable), at the user's request: "aggressively red-team the regating line". Three independent
Opus reviewers read the files (experiments, concept, exposition) and a retrieval agent collected the user's own words
from past sessions; nothing was run. This report is the synthesis. Words as in `GOALS.md` → Terms, except where this
report says a term is broken. Companion plan: `reports/plans/2026-10-28_plan_regating-line-reset.md`. The log's
re-gating passages were rewritten the same day (changelog v88).*

## In plain words

The user suspected three things: that the line was only ever tested on artificial tasks and explains nothing about
real behaviour; that its exposition is confusing; and that nobody has a clear picture of what it is looking for. All
three hold, and the files themselves contain the evidence. The headline number the line was built on, the "switching
share", reads about the same for a fine-tune that learned nothing as for one that learned, so it does not measure what
the update did. The line's one causal test in the real model (E38) ended with Γ's set doing no better than a ranking
that needs no Γ. What survives is narrower and useful: the route split can tell a familiar behaviour from a new one on
tasks where the answer is known, and Γ reads which of the original model's machinery the fine-tuned activations now
look like. Neither is "the update goes into re-gating".

## 1. What "re-gating" actually is (one definition, then the drift)

**The computation.** Take the CI-masked model: the original subcomponents gated by Γ's values, Δ left out. Compute Γ's
values twice, once from the original model's activations (G₀) and once from the fine-tuned model's (G₁). The
*switching share* r is

    r = [L(C₀, G₁) − L(C₀, G₀)] / (T₁ − T₀)

with L the masked model's mean trigger log-odds of the target and T the full model's. In the do-nothing state, which
every re-gating experiment used, the numerator is measured in the masked stand-in and the denominator in the real
model; the update δW sits in Δ, which the stand-in leaves out, so the update reaches the numerator only through G₁.
Code: `src/commute/local_faith.py` (`route_split`); definition: `reports/predictions/2026-10-16_prediction_ladder-routes-amendment.md`.
That prediction file already says (lines 44-45) that "r on its own is not evidence that the components carry the
behaviour", and that in the do-nothing state the masked-off share equals −r exactly. The log never carried that
warning above the cards.

**The drift.** The word is used for at least six objects: route (b) in `GOALS.md` ("where the update goes"); r; the
switched-on set S (alive pieces whose mean trigger importance rises by more than 0.3); the fixed shift to Γ's values
(E18-E36); "the recruited set" and its removal or restore in the real model (E38); and the user's own definition,
"the difference in gating before and after fine-tuning". These are measurements on two different models (the masked
stand-in and the real fine-tuned model) and are not interchangeable. `gaps.html` says re-weighting is undefined and
`overview.html` says a precise version "is not written down yet", while Q5 answers "Yes" to it.

## 2. Does it explain or predict any real behaviour? No.

- **Every re-gating number except E36's is a single-trigger toy.** A 10-step Adam fine-tune of the 67M model forces
  one next token after one trigger token (` :)` → `o`, ` which` → ` was`, ` we` → ` just`, a 32-pair cipher, a
  key-value copy), at about 320 training windows, scored on other occurrences of the same trigger. No experiment
  tested transfer, paraphrase, a capability, or a natural fine-tune's behaviour. E36's lowercase fine-tune was scored
  as log-probability per token; nobody checked that the masked model writes lowercase. No re-gating quantity exists
  on Qwen or GSM8K.
- **No re-gating number predicted a behavioural fact.** Not which inputs change, not off-target damage, not whether
  a fine-tune learns. Reviewer 1 searched every card and report for one and found none.
- **The share is non-specific.** E6: 0.51 to 0.61 for every arm, nulls with another target included. E36's code-text
  fine-tune learned nothing (held-out text 0.05 nats per token worse) and read r = 0.82. E5: r agrees within 0.16
  across the fine-tune, component training and gated-off component training, so where the update lands does not
  move it. The E36 analysis says it: "a high switching share is not evidence of learned re-weighting on its own".
- **The one causal test in the real model is small and beaten.** E38: putting S's trigger outputs back to their
  original values takes back 0.22-0.24 of the gain on emoticon and 0.18-0.20 on ` which` (0.04 on ` we`), against
  about 0 for random, null and already-active sets. That is three to seven times smaller than the switching share
  the line was founded on (0.60, 0.75, 0.28). A set ranked by first-order pull toward the target, with no Γ, matches
  or beats it at the same count. All alive pieces together take back only 0.38-0.49: in the real model the update's
  own write at the trigger carries more than every original piece combined.
- **Replication.** E18, E24 and E38 used two seeds; E28, E29, E32, E33, E35 and E36 ran seed 0 only. One model,
  one position, throughout.

Strongest counter-evidence, stated fairly: E12's route shares separated copy tasks (r 0.76-0.80) from random maps
(0.27-0.29) on a ground truth known by construction, with n = 2 per class; and E38's restore shows Γ's set does pick
pieces the fine-tuned model uses, on two of three toys. Both are instrument results, not mechanism results.

## 3. Is there a clear picture? No, and the architecture forces the headline.

- **Γ is not on the fine-tuned model's causal path.** The fine-tuned model has no Γ. "Route (b)" exists only in a
  hybrid: old pieces, gates computed from another model's activations. `GOALS.md`'s "the change in the new model's
  behaviour splits exactly into five routes" is bookkeeping across three models (masked, unmasked, full); the sum
  telescopes to 1 whatever the mechanism.
- **The reading is forced.** Γ was trained on the original model to turn on a token's machinery when the activations
  look like a context that predicts that token. A fine-tune that makes the model predict the token moves the
  activations toward such contexts (the routing trace's cosines of 0.1-0.4). Γ then emits the original gate pattern
  for that token and the stand-in follows it. The lead's own account on 2026-10-27 concedes: "any change that works
  by cascade has to read as re-gating". A diffuse Adam update therefore reads as re-gating by construction; only a
  concentrated update (E4's six pieces) reads otherwise.
- **The leak was renamed, not ruled out.** The 2026-10-21 decision made "mechanism or leak through Γ's input" the
  first control. E18 defined a leak as *input-specific* information, so a leak of "the answer is `o`" passes as a
  fixed pattern, as "mechanism". On tasks with many answers the fixed pattern collapsed (copy 0.07, cipher 0.15,
  E28) and a per-key change restored it, which is what a leak of the answer predicts; the project read it as "one
  change per cue". The decisive tests (does the shift alone produce the target out of context; a Γ from another VPD
  seed; a Γ refitted on the fine-tuned model) never ran. `claims.html` still says "not a leak".
- **The question changed five times in twelve days** without any version being answered or retired: which route
  dominates (10-16); mechanism or leak (10-21); the form of the shift, H then H′ (10-21/22); its cue-free form (10-23
  to 10-26); whether the real model routes through S (10-27/28). The 2026-10-27 decision listed four readings of
  "understand re-gating" for the user to choose from; all four were the lead's, none the user's, and the lead said in
  chat that "the line had lost its question". After the 67M restore line closed on 2026-10-28, the line has no
  question and no pending run.

## 4. What is worth keeping

1. **The route split as a calibrated instrument**, not as a mechanism claim. On anchors whose nature is known by
   construction (copy or distilled-prompt behaviour = existing; invented maps or facts = new) it separates the two
   (E12). Read unlabelled on a real update, with the nulls that already broke the share built in (another-target
   update, a non-learning update, a random δW matched on output change), it bears on `GOALS.md`'s fork "new
   capability or strengthened existing one". This is the only non-circular signal in the record.
2. **Γ as a target-free detector of which original machinery the fine-tuned activations now resemble** (E38's
   census: 60-96% of S is the target's own Pile machinery). Useful as a *proposal* of a set to test causally in the
   real model, never as the finding.
3. **The saved states and code**: E18's do-nothing states and recruited sets, `recruitment.py`'s restore, the routing
   trace. They make the nulls above cheap ($0 forward passes on the cluster).

## 5. What should stop

- Writing that the update "goes into" or "is carried by" re-gating, and "reveal reads as r". The share is a
  property of the stand-in and of Γ's generalisation.
- The fixed-shift block (E18-E36) as anything but a closed note.
- Any new 67M toy-trigger re-gating run. The user's words (2026-10-27): "the experiments you have run have been so
  artificial they dont really tell us anything general"; "anything that depends too strongly on defining cues or
  triggers is bad".
- Pending cards that offer wording choices. The user: "it is difficult to tell what you are stuck on and why you need
  input from me".

## 6. Exposition defects found (fixed in the same day's flush; detail in the exposition audit)

The three reviewers' inventories agree on the pattern: the same word for several objects; numbers given without
what they measure; conclusions stated as premises (Q5 opens "Yes"); stale passages after the 2026-10-27 and
2026-10-28 closures; the warning in the prediction file (r alone is no evidence) absent from every section above
the cards; "recovery" defined one way in the glossary and another in the code; `techniques.html` describing the
shares as taken in sequence where the prediction file says no order is chosen. The log's re-gating passages
(Overview Line 1 box, Q5, the techniques paragraph, the Next card, glossary entries, claims rows, Gaps) were
rewritten to one definition, one story and one open question; the old wording is in git.

## Not run

Nothing. Every number above is quoted from an existing report, card or code file; the reviewers' full reports are
in this session's transcript and their file citations are reproduced in the plan.
