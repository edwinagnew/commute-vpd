# Options: one real task for the Qwen comparison, with variants as controls

*2026-11-06, qwen-rl-lead, for the user, who wrote (task file): "all of these options seem a bit synthetic ... a single
real-world task as the main experiment and various modifications of it as controls ... push back as needed". Replaces
the options in `2026-11-06_decision_anchor2-qwen.md`; nothing runs until the user picks. Words as in `GOALS.md` → Terms.*

**In plain words.** The user is right that the comparison table has become a set of small invented tasks, each built
to answer a separate question. The cleaner design takes one real task, the maths fine-tunes already trained for the
SFT-against-RL table, and makes the "already known" and "new" examples out of that same task, sorting its problems
by whether the unchanged model can already solve them. One real cost: on real data, novelty is measured rather than
built in, so the "new" label is weaker than the invented-facts version and has to be stated as "the unchanged model
cannot do it at our sampling budget".

## The pushback first

- The synthetic tasks exist for one reason: an anchor's kind has to be known before the readings see it. With real
  data the kind is *measured* (the unchanged model's success when sampled many times), so "new" means "not found in
  k samples", never "absent". Recombining skills it has would still read as elicited to VPD (the granularity limit,
  user 2026-10-29). I think that is acceptable if said plainly; it is also how the literature argues the RL question
  (Yue et al. 2025, pass@k of base against RL-trained models).
- Real "new" updates end larger than "known" ones. Every option below compares at a matched update size (an earlier
  step of the larger run), as the current plan already does.
- GSM8K alone is too easy: Qwen3-1.7B solves nearly every problem if sampled enough, so there is almost no "new" pool.
  Options 1 and 2 add the MATH set's harder problems, same format.

## Options

**1 (recommended). One maths task, problems sorted by the unchanged model's pass rate.** Sample the unchanged model
64 times per problem on GSM8K plus MATH levels 3-5. Known pool: problems solved often. New pool: problems never solved,
matched on level and topic. Fine-tune (SFT on the gold worked solutions, same recipe as the D2 gold arm) on each pool;
the existing GSM8K GRPO, SFT, shuffled-reward and template states are the main experiment beside them.
- Pros: one task, one format, real data; it merges the comparison table into the SFT-against-RL work, so the maths
  arms get read against anchors from their own task; labels come from a measurement the field already uses; ~$0 on
  the cluster (sampling is a few L40 hours, fine-tunes ~6 min each).
- Cons: "new" is soft (see above); the never-solved pool may need skills the model partly has; GRPO cannot train on
  never-solved problems (no reward), so the new pool is SFT only, and RL is compared only on the known side.

**2. Same as 1, but the control is a variation of the problems, not a different pool.** Take known-pool problems and
change one thing the model has not seen: a different number base for the answers, or a unit system or rule defined
in the prompt. Known arm: the originals.
- Pros: tighter pairing (same problems), so novelty is the only difference by construction.
- Cons: this is the "synthetic" move again, just on real problems; a rule given in the prompt is closer to following
  instructions than to learning something new.

**3. Real facts the model cannot know: events after its training cut-off.** Question-answer pairs from 2025 Wikipedia
articles (new) against the same kind of article from before the cut-off that the model answers correctly (known),
same template; a novelty check on the unchanged model first.
- Pros: the most realistic "teaches something new" update there is (keeping a model current); same source and format
  on both sides.
- Cons: a second task beside maths, not a variant of it, so it does not give the single-task design you asked for;
  needs a new dataset build (~1 day); facts, not a skill.

**4. A real skill graded by how much the model has: translation into a language it barely knows.** FLORES-200,
translation into a low-resource language (new) against French or Spanish (known), novelty from the unchanged model's
score.
- Pros: a real benchmark; a capability, not facts.
- Cons: a 1.7B model knows a little of most languages, so new and known differ by degree; learning a language from a
  small fine-tune is slow, so the new update ends far larger.

**Not recommended: the real base-to-instruct update** (Qwen3-1.7B-Base to Qwen3-1.7B). The most organic post-training
update there is, but it is far larger than a small update (not measured here) and mixes everything, so no reading can be
assigned to elicitation or novelty.

## What happens to what exists

The template fine-tune and the invented-authors (TOFU) fine-tune are already trained and read; they stay as extra
columns at no cost, labelled as the earlier synthetic anchors. Under option 1 they become side checks, not the anchors.

## If option 1 is chosen

`/design` check, then: sampling job (pass rates, cluster), pool build with level and topic matching, two SFT arms on
two seeds, update saved at the matched size and at the end; `qwen-decomp-lead` reads the three readings on them. The
schedule's 11-11 midpoint is tight: sampling 11-07, fine-tunes 11-08, readings 11-09 to 11-10.

## Not run

Nothing. The pass-rate split of GSM8K and MATH for Qwen3-1.7B is not yet measured, so the size of the "new" pool is a
guess: the first job would measure it, and option 1 falls back to MATH level 5 only if the pool is under ~100 problems.
