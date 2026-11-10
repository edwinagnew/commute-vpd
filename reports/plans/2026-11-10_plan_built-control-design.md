# A "built" control that does not rest on linear unreadability alone

*2026-11-10. Cell `P1.8`, design only: nothing here has run, no experiment number is claimed, and a run needs you.
The wording read is `~/commute-data/scratch/definition-session/2026-11-09_reuse-at-a-variable_v2.md`, unchanged.*

**In plain words.** Under the second wording, "built" means that no affine map of the original's activations sets
the changed variable, so any built reading is, in the end, a test of linear unreadability. A built label known "for
another reason" can only come from how the original was made. The reading then tests something only when the same
quantity is read against two originals, one given the skill and one not. I propose one
such pair on the 67M, cheap and ready to write, and one on small models pretrained with and without a skill.

## Design check

- **Goal served.** Part 1 of the paper plan: the gate page needs a built control as designed or run.
- **Nearest tried.** The hand-made XOR pair of the first test, whose built reading follows from its construction.
  The counting-grammar models of the reuse-anchor study set possession by pretraining mix, but a count cannot serve:
  see the second "for the user" line.
- **Patch or step.** A patch on the first test, asked for by its premise check.
- **Decisions relied on.** The wording as written and the cut-offs filed for the fine-tune test: G 0.794, H 0.476.
  No new rule, band or clause.
- **Control first.** In each candidate, the original that was given the skill is checked first, alone, for an affine
  read of the quantity above 0.55 on held-out windows. If that fails, the pair cannot serve, and the job stops.

## Why a contrast, not a single pair

The fine-tune test's filed rule reads an H below its cut-off as "no map found", never as built, because a search
can always miss. A single built pair can therefore never read built under the rules now filed. A matched pair fixes
this. Suppose the same search, on the same quantity and inputs, finds the map when the original had the skill. Then
its failure on the original without it is about that original, not about the search or the quantity. That is what the XOR pair lacked.

## Candidate 1: the same XOR, two originals of the 67M

**Construction.** Two originals. The plain one is the 67M itself. The taught one is the 67M after a short
fine-tune that imitates the first test's XOR pair at one third of its strength, on every window. Each original is
then fine-tuned, as in the fine-tune test, to imitate the full XOR pair. The quantity, the inputs, the target
behaviour, the distance and the search are the same for both. Only the original differs.

**Why the labels hold.** The taught original shows the XOR behaviour on every input, so it computes the XOR by how
it was made. Its fine-tune only turns that up, which is the wording's own "turned up" row: re-weighted, or rewired
if the change is spread too widely for four directions. The plain original never received the seeded bits, drawn for
the first test after the 67M was trained, so its weights carry no information about them. Its built label rests on
that and on the first test's capacity check, 0.506 against 0.502 for shuffled bits. That arm is not new: it is the
fine-tune test's XOR imitator. What is new is the taught arm, the positive control matched on the quantity.

| Reading | Taught original's fine-tune | Plain original's fine-tune |
|---|---|---|
| Affine read of the XOR bit from the original, bar 0.55 | Well above, by construction | 0.506, measured |
| H, cut-off 0.476 | Above, at least at the last sites | Below, as the XOR pair read 0.009 below zero |
| G, cut-off 0.794 | Above if the change stays in a few directions; may fall below, as the first fine-tune of the "Always on" pair read under 0.45 at the sites logged so far | Below |
| Label by the filed rule | Re-weighted or rewired | No map found |

**Cost.** $0 on the cluster's debug queue, about 2.5 hours in two jobs. The taught original takes about two minutes
to train. One fine-tune plus its read takes 35 to 45 minutes. The first test's four pairs took about 2.2 hours
across jobs 57142 and 57208, and the fine-tune test's G search alone took about 22 minutes per item in job 57673.
Three seeds of the taught arm, plus the control. The plain arm's three seeds are the fine-tune test's XOR items,
queued now. Code: a short run script on `src/commute/variable_read.py`, which already reads a fine-tuned copy.

**What each outcome changes on the gate page.**
- *Taught above 0.476 on H, plain below.* The built row gains a control whose result can come only from the
  original. On this quantity the low H tracks what the original had. The search was shown to succeed on the same
  quantity, so its "no map found" carries weight.
- *Both below.* The search cannot find the XOR even where the original computes it. The built row says the
  wording is untested against a built case, and low H means nothing on this quantity.
- *Both above.* The plain fine-tune found a route through what the 67M already reads. The gate page records that an
  ordinary fine-tune of the 67M never produced a built case.
- *The control fails.* The taught original does not hold the XOR linearly although it shows the behaviour. That
  reads against the stipulation itself and goes to you before anything else.

## Candidate 2: small models pretrained with and without a skill

**Construction.** Two of the reuse-anchor study's grammar models: six blocks, width 192, about 3M parameters, 50K
pretraining steps. Both use its "counts b only" mix, so neither counts a. Both also see windows under a new prefix Q
asking whether the count of a is even. In the taught model Q's answers are true; in the plain model they are drawn
at random, so both see the same tokens. Each is then fine-tuned to answer the same question after a second prefix P,
which carried random answers in both pretrainings.

**Why the labels hold.** The plain model had no training signal for the count of a or its parity, so built comes
from its history, not from a chosen hard function. Unlike the XOR, whether parity is affinely readable from it is
open. The study's "counts b only" model was never asked for the count of a, yet a probe read it at 0.44 to 0.52. The taught model computes parity under Q by how it was made.

| Reading | Taught | Plain |
|---|---|---|
| Affine read of parity from the original, bar 0.55 | Above, checked first | Open: this is the test |
| H, cut-off 0.476 | Above | Below if history and readability agree |
| G, cut-off 0.794 | Above if P is wired to Q's existing answer | Below |

**Cost.** $0, two or three debug jobs. The study's plan budgeted three such pretrainings at one to two hours in all,
so two take under an hour and a half. Fine-tunes take minutes on a 3M model. Code is the larger cost: the read code
assumes the 67M's sites, so a writer must port it to the grammar model, about a day. The cut-offs were set on the
67M's hand-made pieces; using them on another model is a departure, so this pair is read as a contrast only.

**What each outcome changes on the gate page.**
- *Taught high, plain low.* A second, independent built control, on a pretraining history rather than a fine-tune.
- *Both high.* The wording calls a skill the original was never trained on "rewired", because parity of a count is
  affinely readable from features it learned for other reasons. The gate page lists this under where the stipulation
  and the intended meaning part, for you to accept or reject.
- *Both low.* As in candidate 1: the search fails on this quantity.

## Recommendation

Candidate 1 first, after the fine-tune test's XOR items have read: it costs under three debug hours, reuses the code
that is already running, and answers the premise check directly. Candidate 2 only if you want the gate page to say
something about pretraining histories, where it is the only one of the two that can disagree with the stipulation.

## For the user

1. Under the wording's stipulation, every built label is a statement of linear unreadability. A control can give
   "built" another source only through the original's history, and a run reads it only as a contrast.
2. A new route to a quantity the original reads linearly is rewired by the wording's own rewired row, so it cannot
   be a built control. The same holds for any count of a token: since the original's activations include the
   embedding at every position, a count is affine in them for any original, trained or not.
3. The filed rule never reads "built" from a low H. With the contrast above, a low H could be read as built, but that
   reading is a rule only you can add.

## Departures from the literature

Jain et al. 2024 set possession by pretraining mix, as candidate 2 does. They read it with a probe on one layer and
position; here it is the wording's affine map from every earlier layer and position. Otherwise none beyond the first
test's list.

## Not run

Both candidates. The fine-tune test's XOR items, still queued. The taught original's capacity check. Porting the read
code to the grammar model. Any check that the public with-and-without models the wording lists exist and load.
