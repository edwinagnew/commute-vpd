# Review: can anchor 2 be read on the KL grid? (E45, advice, 2026-10-29)

*Fable, for the qwen-decomp lead. Read: the re-plan §5/§7, the Q2 plan §3/§5/§8, Amendment 1, `results/{T,E}_a2long_s{0,1}.json`,
`E_gold_s0.json`, `score.json`, `control.json`, `predictions*.json`. Advice only; the user decides.*

**Plain words.** Anchor 2 now learns its facts (held-out exact match 0.91 at step 100, 1.0 at step 300, both seeds),
but every point of the filed KL grid sits at steps 2-12, where it recalls 1-2% of them, the same as W₀. That is not
a recipe failure: for a content anchor, own-row KL *is* the learning (each new fact moves its answer tokens from
near-zero to near-one probability, many nats each), so "learned at own-KL 0.1" is a contradiction, not a target. On
the shared maths axis the movement saturates at ~0.6 by step 30-40 (tracker), while facts are still being fit to step
100: maths-KL cannot pick a learned state either. Under `predictions.json` as filed this is the listed edge case "an
anchor learned at no grid point: row unscored, card", so the decisive rows are unscored for anchor 2 unless a new
point is filed. GSM8K's fall to 0.13-0.20 looks like a format transfer (responses 198 → 18-47 tokens, the one-line
"The answer is X" style), not necessarily lost maths; it starts before any fact is learned (length 152 at step 12).

## 1. Options for the grid

| option | cost | verdict |
|---|---|---|
| (a) Extend the grid upward on both axes and read anchor 2 where learned | own axis: learned needs own-KL ≥ ~4, no other arm will ever be there (gold reaches 0.86 at step 100); maths axis: saturates at 0.6 before learning, so a higher grid value does not exist to add | not possible as a KL point |
| (b) Lower the learned threshold | exact match is 0.013-0.022 at every grid state against W₀'s 0.015/0.008; no threshold above chance helps. A held-out *loss* criterion would fire at step 3-6 (2.93 → 1.63 nats), but that is the template tokens being learned, not facts | dead |
| (c) Change the recipe (fewer facts, replay, LoRA) | fewer facts: per-token own-KL per learned fact is unchanged. Replay of maths data: would cut the maths-KL and the GSM8K collapse but puts maths rows into the content anchor's δW, breaking the domain-confound sentence the other way. LoRA: a second update type, against the one-recipe rule. Each is the last amendment plus ~3 h cluster, $0 | keep in reserve for §2 |
| (d) Read anchor 2 at its first learned saved state (step 100: own-KL 4.5/4.7, maths-KL 0.64/0.61 by seed) and match the *other* arms to it on the shared maths axis, matched KL stated | $0, states on disk; gold's step 100 is already at maths-KL 0.70 (|log ratio| 0.1, inside the grid's own bracket spacing). A new filing in Amendment 2 (the last), or in Q3's filing for the rows that arrive there (route split) | **favoured** |

The plan's logic favours (d): rule §5.4 asks for matching on KL, not steps, and the only axis shared by all arms is the
maths rows (own rows differ per arm, so own-axis matching already compares different quantities). The filed decisive
point (own-KL 0.1) stands and is reported as unreachable for anchor 2; Amendment 2 files anchor 2's decisive state
as "first saved learned state, matched on the maths axis within a stated tolerance (say a factor 1.5)". The honest
limit: an arm that never reaches maths-KL ~0.6 (own samples on policy may not) is not matched, and that is itself a
finding to report (a content update costs more movement than the GSM8K updates make), with the step-100 fallback the
plan already names ("unmatched, not used to validate a row").

## 2. The GSM8K collapse

It confounds the maths-axis readings, not the own-axis ones: anchor 2's δW on maths rows is nearly all a format shift
(a behaviour the model produces under a prompt, anchor 1's kind of novelty), so behaviour-kept or placement read on
maths rows would score anchor 2 as anchor-1-like. On its own rows the content is present (exact match 0.999) and the
format is the one those rows use. Two $0 checks separate damage from format, each one E-style cluster job of ~10 min,
no amendment (a reading, not an arm): (i) GSM8K test 0-255 at steps 100 and 300 under the stage-P 4-shot prefix or a
"show your steps" instruction, paired against W₀ under the same prompt: recovery to ~0.78 means the maths machinery
is intact and the collapse is format; (ii) teacher-forced loss on E40's 256 held-out gold solutions at the same
states against W₀ (a rise of a few hundredths means intact; tenths means damage). Also note seed 1's known facts dip
to 0.595 at step 12 before recovering: early anchor-2 states are not merely "nothing learned" but transiently damaged.

## 3. Recommendation, and what would change it

Spend nothing on the recipe now. Run checks (i) and (ii) as FYI; file Amendment 2 as (d), with anchor 2 read at step
100 on its own rows for the decisive rows and every maths-row reading of anchor 2 carrying the sentence "anchor 2's
maths-row movement is a format shift, not its content". **Would change it:** if (i) does not recover (accuracy
stays under ~0.5 with the prompt) the learned state is content plus damage, and the last amendment should go to the
recipe instead: drop the one-line format request from anchor 2's prompt (the likely cause) rather than add replay.
If own samples never reach maths-KL ~0.4 by step 100, report the unmatched readings and do not chase a match.

## Not run
Checks (i) and (ii); no state between steps 12 and 100 exists, so where exactly recall crosses 0.25 is unknown
(tracker: training loss 1.26 at step 12, 0.46 at step 50, 0.03 at step 100).
